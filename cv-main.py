import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import io
import json

# --- 页面基础配置 ---
st.set_page_config(
    page_title="CV 数据高级绘图 (支持 .pssession)",
    page_icon="📈",
    layout="wide"
)

# --- 核心逻辑：通用数据提取 ---
def extract_values_from_array(data_array):
    """
    从可能是 [1, 2, 3] 或 [{"V":1}, {"V":2}] 的数组中提取数值
    """
    if not data_array or not isinstance(data_array, list):
        return []
    
    # 检查第一个元素
    first = data_array[0]
    if isinstance(first, (int, float)):
        return data_array
    elif isinstance(first, dict) and "V" in first:
        # 提取字典中的 "V" 值 (PalmSens 序列化格式)
        return [item.get("V", 0) for item in data_array]
    return []

def find_nested_data(curve_obj, axis_names):
    """
    在 curve 对象中递归或按路径寻找指定轴的数据
    """
    # 1. 尝试直接获取 (x, xValues, XAxisDataArray...)
    for key in axis_names:
        if key in curve_obj:
            val = curve_obj[key]
            # 情况A: 直接是列表
            if isinstance(val, list):
                extracted = extract_values_from_array(val)
                if extracted: return extracted
            # 情况B: 是对象，里面包含 m_values (常见于 .pssession)
            elif isinstance(val, dict) and "m_values" in val:
                extracted = extract_values_from_array(val["m_values"])
                if extracted: return extracted
            # 情况C: 是对象，里面包含 values
            elif isinstance(val, dict) and "values" in val:
                extracted = extract_values_from_array(val["values"])
                if extracted: return extracted
                
    return []

# --- 核心逻辑：解析 .pssession (JSON) ---
def parse_pssession(file):
    """
    解析 PalmSens .pssession (JSON格式) 文件
    支持多种 JSON 结构变体
    """
    datasets = {}
    try:
        # 1. 获取文件内容并解码
        content = file.getvalue().decode('utf-8', errors='ignore')
        
        # 2. 循环解析所有 JSON 对象 (修复 Extra Data 错误)
        decoder = json.JSONDecoder()
        pos = 0
        all_json_objects = []
        
        while pos < len(content):
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            try:
                obj, end_pos = decoder.raw_decode(content, idx=pos)
                all_json_objects.append(obj)
                pos = end_pos
            except json.JSONDecodeError:
                break
        
        # 3. 提取数据
        for data_json in all_json_objects:
            measurements = []
            # 寻找 measurements 列表
            if isinstance(data_json, dict):
                if "measurements" in data_json: measurements = data_json["measurements"]
                elif "Measurements" in data_json: measurements = data_json["Measurements"]
                elif "curves" in data_json or "Curves" in data_json: measurements = [data_json] # 根节点即 measurement
            
            for m_idx, meas in enumerate(measurements):
                if not isinstance(meas, dict): continue
                title = meas.get("title", meas.get("Title", f"Meas"))
                curves = meas.get("curves", meas.get("Curves", []))
                
                for c_idx, curve in enumerate(curves):
                    # --- 智能搜索 X 和 Y 数据 ---
                    # 定义可能的键名优先级
                    x_keys = ["x", "xValues", "X", "XAxisDataArray", "x_values"]
                    y_keys = ["y", "yValues", "Y", "YAxisDataArray", "y_values"]
                    
                    x = find_nested_data(curve, x_keys)
                    y = find_nested_data(curve, y_keys)
                    
                    if len(x) > 0 and len(y) > 0:
                        # 确保长度一致
                        min_len = min(len(x), len(y))
                        x = x[:min_len]
                        y = y[:min_len]
                        
                        # 构建唯一名称
                        clean_fname = file.name.rsplit('.', 1)[0]
                        name = f"{clean_fname}"
                        if len(measurements) > 1 or len(curves) > 1:
                            name += f"-{title}"
                        if len(curves) > 1:
                            name += f"-C{c_idx+1}"
                            
                        df = pd.DataFrame({'V': x, 'I': y})
                        datasets[name] = df
                    
    except Exception as e:
        st.error(f"解析 .pssession 文件 {file.name} 失败: {str(e)}")
        
    return datasets

# --- 核心逻辑：解析 CSV/Excel (保持不变) ---
def parse_spreadsheet(file):
    filename = file.name
    if filename.endswith('.csv'):
        df_raw = pd.read_csv(file, header=None)
    else:
        df_raw = pd.read_excel(file, header=None)

    datasets = {}
    row0 = df_raw.iloc[0].values
    
    for i in range(0, df_raw.shape[1], 2):
        if i + 1 >= df_raw.shape[1]: break
        name = str(row0[i]).strip()
        if name in ['nan', '', 'None']: name = f"Sample_{i//2 + 1}"
        
        base_name = name
        counter = 1
        while name in datasets:
            name = f"{base_name}_{counter}"
            counter += 1
            
        sub_df = df_raw.iloc[2:, i:i+2]
        sub_df.columns = ['V', 'I']
        sub_df = sub_df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if not sub_df.empty:
            datasets[name] = sub_df
    return datasets

# --- 主界面逻辑 ---
st.title("🔬 电化学 CV 数据对比与绘图")

# 侧边栏
with st.sidebar:
    st.header("1. 数据上传")
    uploaded_files = st.file_uploader("选择文件", type=['csv', 'xlsx', 'xls', 'pssession', 'json'], accept_multiple_files=True)
    
    st.header("2. 绘图设置")
    font_family = st.selectbox("字体", ["Arial", "Times New Roman", "Helvetica"], index=0)
    font_size = st.slider("字号", 10, 24, 14)
    line_width = st.slider("线宽", 0.5, 4.0, 2.0)
    
    st.subheader("数据单位处理")
    # PalmSens .pssession 通常是 V 和 A。用户可能需要转换。
    current_mult = st.selectbox("电流乘数", 
                               [1, 1e3, 1e6], 
                               index=2, 
                               format_func=lambda x: "x1 (A)" if x==1 else ("x10³ (A→mA)" if x==1e3 else "x10⁶ (A→µA)"))
    
    potential_mult = st.selectbox("电位乘数", 
                                [1, 1e-3], 
                                index=0,
                                format_func=lambda x: "x1 (V)" if x==1 else "x10⁻³ (mV→V)")

    st.subheader("坐标轴")
    x_label = st.text_input("X 轴标签", "Potential (V)")
    y_label = st.text_input("Y 轴标签", "Current (µA)")
    reverse_x = st.checkbox("翻转 X 轴", value=False)
    reverse_y = st.checkbox("翻转 Y 轴", value=False)

# 处理数据
all_datasets = {}
if uploaded_files:
    for f in uploaded_files:
        f.seek(0)
        fname = f.name.lower()
        if fname.endswith(('.pssession', '.json')):
            d = parse_pssession(f)
        else:
            d = parse_spreadsheet(f)
        if not d:
            st.warning(f"⚠️ {f.name}: 未识别到有效曲线数据。")
        all_datasets.update(d)

# 绘图
if all_datasets:
    st.header("数据选择")
    selected_names = st.multiselect("选择曲线", list(all_datasets.keys()), default=list(all_datasets.keys())[:2])
    
    if selected_names:
        # 配色
        cols = st.columns(min(len(selected_names), 5))
        colors = {}
        palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']
        for idx, name in enumerate(selected_names):
            with cols[idx % len(cols)]:
                colors[name] = st.color_picker(name, palette[idx % len(palette)])
        
        # Matplotlib 绘图
        mpl.rcParams['font.family'] = font_family
        mpl.rcParams['font.size'] = font_size
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
        
        for name in selected_names:
            df = all_datasets[name]
            
            # 智能单位处理
            # 只有当数据看起来非常小（像安培）时，才应用乘数
            # 或者如果用户强制选择了乘数，就应用
            
            # 电位处理
            x_data = df['V'] * potential_mult
            
            # 电流处理
            y_data = df['I'] * current_mult
                
            ax.plot(x_data, y_data, label=name, color=colors[name], linewidth=line_width)
            
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        if reverse_x: ax.invert_xaxis()
        if reverse_y: ax.invert_yaxis()
        
        ax.legend(frameon=False)
        ax.tick_params(top=True, right=True)
        
        st.pyplot(fig)
        
        # 导出
        col1, col2 = st.columns(2)
        pdf = io.BytesIO()
        fig.savefig(pdf, format='pdf', bbox_inches='tight')
        col1.download_button("下载 PDF", pdf.getvalue(), "cv.pdf", "application/pdf")
        
        png = io.BytesIO()
        fig.savefig(png, format='png', dpi=300, bbox_inches='tight')
        col2.download_button("下载 PNG", png.getvalue(), "cv.png", "image/png")

else:
    st.info("请上传数据文件开始。")
