import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import sys
import sklearn.compose._column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# Check dependencies
try:
    from catboost import CatBoostRegressor
except ImportError:
    st.error("Please run in terminal: pip install catboost")

# ==============================================================================
# 1. Model Patch & Wrapper (修复了Imputer不工作导致GBR接收NaN崩溃的问题)
# ==============================================================================
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __repr__(self): return super().__repr__()
        def __getstate__(self): return self.__dict__
        def __setstate__(self, d): self.__dict__.update(d)
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

class SafeKNNImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, noise_level=0.01, random_state=42):
        self.n_neighbors = n_neighbors
        self.noise_level = noise_level
        self.random_state = random_state
        self.imputer_ = None

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        # 核心修复：真正调用模型内部保存好的 imputer_ 来填补 NaN！
        if self.imputer_ is not None:
            X_imputed = self.imputer_.transform(X)
            # 保持数据格式一致
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            return X_imputed
        
        # 如果模型里没有 imputer (理论上不会发生)，作为最后防线把 NaN 转成 0
        return np.nan_to_num(X, nan=0.0)

# 把类注册到 __main__ 空间，防止 joblib 加载时找不到
sys.modules['__main__'].SafeKNNImputerWrapper = SafeKNNImputerWrapper

def repair_model_attributes(obj):
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype"): obj._fill_dtype = np.float64
        if not hasattr(obj, "_parameter_constraints"): obj._parameter_constraints = {}
    if hasattr(obj, 'steps'):
        for name, step in obj.steps: repair_model_attributes(step)
    if hasattr(obj, 'transformers_'):
        for name, trans, cols in obj.transformers_: repair_model_attributes(trans)
    if hasattr(obj, 'imputer_') and obj.imputer_ is not None:
        repair_model_attributes(obj.imputer_)
    return obj

st.set_page_config(page_title="Material Property Prediction", layout="wide")

# ==============================================================================
# 2. Extract mappings from Excel automatically
# ==============================================================================
@st.cache_data
def load_excel_mappings():
    file_path = '输入输出说明.xlsx'
    if not os.path.exists(file_path):
        return {'MOx': {'Al2O3(Missing Excel)': 0}, 'Sys.': {'Cubic': 7}, 'S.G.': {'P1': 1}}
    try:
        df = pd.read_excel(file_path, header=None)
        def extract_map(n_col, v_col):
            mapping = {}
            for r in range(5, len(df)):
                name = str(df.iloc[r, n_col]).strip()
                val = df.iloc[r, v_col]
                if name and name.lower() != 'nan' and str(val).lower() != 'nan':
                    try: mapping[name] = int(float(val))
                    except ValueError: pass
            return mapping
        return {
            'MOx': extract_map(8, 9),
            'Sys.': extract_map(11, 12),
            'S.G.': extract_map(14, 15)
        }
    except Exception as e:
        return {'MOx': {'Error': 0}, 'Sys.': {'Error': 7}, 'S.G.': {'Error': 1}}

# ==============================================================================
# 3. Load models
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    'Cp_A': 'Cp_A_GBR_模型_安全版.pkl',
    'Cp_B': 'Cp_B_GBR_模型_安全版.pkl',
    'Cp_C': 'Cp_C_CatBoost_模型_安全版.pkl',
    'Cp_D': 'Cp_D_CatBoost_模型_安全版.pkl',
    'G0':   'G0_GBR_模型_安全版.pkl'
}

@st.cache_resource
def load_and_fix_models():
    loaded = {}
    for target, filename in MODEL_FILES.items():
        path = os.path.join(CURRENT_DIR, filename)
        if os.path.exists(path):
            try:
                loaded[target] = repair_model_attributes(joblib.load(path))
            except Exception as e:
                pass
    return loaded

# ==============================================================================
# 4. Main UI
# ==============================================================================
def main():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if not st.session_state.get('logged_in', False):
        st.title("🔒 System Login")
        u = st.text_input("Username", "admin")
        p = st.text_input("Password", "123456", type="password")
        if st.button("Login"):
            if u == "admin" and p == "123456":
                st.session_state['logged_in'] = True
                st.rerun()
            else: st.error("Incorrect username or password")
        return

    st.title("🧪 Material Property Prediction System")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False))

    CAT_MAPPINGS = load_excel_mappings()
    models = load_and_fix_models()

    FEATURE_CATEGORIES = {
        "Elemental Intrinsic Properties": ['MOx', '1-Z', '2-Z', '3-Z', '1-nv', '2-nv', '3-nv', '1-O.S.', '2-O.S.', '3-O.S.', '1-rcov', '2-rcov', '3-rcov', '1-I1', '2-I1', '3-I1', '1-Χp', '2-Χp', '3-Χp'],
        "Structural and Electronic Characteristics": ['Sys.', 'S.G.', 'L a', 'L b', 'L c', 'L α', 'L β', 'L γ', '1-rm', '2-rm', '3-rm', 'rm', '1-rs', '2-rs', '3-rs', 'rs', '1-rmin', '2-rmin', '3-rmin', '1-rmax', '2-rmax', '3-rmax', 'Eg', 'M/O'],
        "Thermodynamic Properties": ['1-S0', '2-S0', '3-S0', 'Tmin', 'Tmax', 'T']
    }

    input_data = {}        
    readable_data = {}     
    
    tabs = st.tabs(list(FEATURE_CATEGORIES.keys()))
    
    for i, (cat, features) in enumerate(FEATURE_CATEGORIES.items()):
        with tabs[i]:
            cols = st.columns(3)
            for j, f in enumerate(features):
                with cols[j%3]:
                    if f in CAT_MAPPINGS:
                        opts = ["--- Optional ---"] + list(CAT_MAPPINGS[f].keys())
                        sel = st.selectbox(f"Select {f}", opts, key=f"sel_{f}")
                        
                        if sel == "--- Optional ---":
                            input_data[f] = np.nan
                            readable_data[f] = "Blank"
                        else:
                            input_data[f] = CAT_MAPPINGS[f][sel]
                            readable_data[f] = sel
                    else:
                        val_str = st.text_input(f"Input {f}", value="", placeholder="Leave blank", key=f"num_{f}")
                        if val_str.strip() == "":
                            input_data[f] = np.nan
                            readable_data[f] = "Blank"
                        else:
                            try:
                                num_val = float(val_str)
                                input_data[f] = num_val
                                readable_data[f] = num_val
                            except ValueError:
                                st.error(f"Invalid number: {val_str}")
                                input_data[f] = np.nan
                                readable_data[f] = "Invalid"

    st.divider()

    # ==============================================================================
    # 5. Prediction
    # ==============================================================================
    if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
        df_raw = pd.DataFrame([input_data])
        st.subheader("Prediction Results:")
        res_cols = st.columns(5)
        targets = ['Cp_A', 'Cp_B', 'Cp_C', 'Cp_D', 'G0']
        
        current_record = readable_data.copy()
        
        for idx, t_name in enumerate(targets):
            model = models.get(t_name)
            with res_cols[idx]:
                if model:
                    try:
                        if hasattr(model, "feature_names_in_"):
                            expected = list(model.feature_names_in_)
                            X_input = pd.DataFrame(index=[0])
                            for col in expected:
                                X_input[col] = df_raw.get(col, np.nan)
                            X_input = X_input[expected].astype(float)
                        else:
                            X_input = df_raw.astype(float)
                        
                        pred = model.predict(X_input)[0]
                        st.metric(label=t_name, value=f"{pred:.4f}")
                        current_record[t_name] = round(pred, 4)
                        
                    except Exception as e:
                        # 把具体的报错打印出来
                        st.error("Error")
                        st.caption(f"{str(e)[:80]}") 
                        current_record[t_name] = "Error"
                else:
                    st.metric(label=t_name, value="Not Ready")
                    current_record[t_name] = "Not Calc."
        
        st.session_state['history'].append(current_record)

    # ==============================================================================
    # 6. History
    # ==============================================================================
    if st.session_state['history']:
        st.divider()
        st.subheader("📊 Prediction History")
        
        df_history = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_history, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            csv_data = df_history.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 Export to CSV", data=csv_data, file_name="Prediction_History.csv", mime="text/csv", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state['history'] = []
                st.rerun()

if __name__ == "__main__":
    main()