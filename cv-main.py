import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import io
import json
import re

# --- 页面基础配置 ---
st.set_page_config(
    page_title="CV 数据分析与绘图 (全能版)",
    page_icon="⚡",
    layout="wide"
)

# --- 核心工具 1: 处理 PalmSens 复杂的列表结构 ---
def extract_values_from_complex_list(data_list):
    """
    专门处理 PalmSens 的数据列表。
    PalmSens 的数据可能是:
    1. 简单数组: [0.1, 0.2, 0.3]
    2. 对象数组: [{"V": 0.1, "C": 1}, {"V": 0.2, "C": 1}] 
       (注意: 这里的 "V" 代表 Value 数值，不一定代表 Voltage)
    """
    if not data_list or not isinstance(data_list, list):
        return []
    
    if len(data_list) == 0:
        return []

    first_item = data_list[0]
    
    # 情况 A: 列表里直接是数字
    if isinstance(first_item, (int, float)):
        return data_list
    
    # 情况 B: 列表里是字典对象
    elif isinstance(first_item, dict):
        # 优先提取 'V' (Value) 键
        if "V" in first_item:
            return [item.get("V", 0) for item in data_list]
        # 备选 'y' 或 'v'
        elif "y" in first_item:
            return [item.get("y", 0) for item in data_list]
            
    return []

# --- 核心工具 2: 智能识别 X 和 Y 轴 ---
def smart_find_axis_data(curve_obj):
    """
    在 Curve 对象中遍历所有属性，通过 Unit (单位) 元数据来寻找 X 和 Y。
    """
    x_candidates = []
    y_candidates = []

    # 遍历 Curve 下的所有属性
    for key, val in curve_obj.items():
        # 我们只关心字典(复杂数据)或列表(简单数据)
        if not isinstance(val, (dict, list)):
            continue

        raw_data = []
        axis_type = "unknown" # potential, current

        # --- 分支 1: 属性是字典 (包含 m_values 和 Unit) ---
        if isinstance(val, dict):
            # 提取数据部分
            if "m_values" in val:
                raw_data = val["m_values"]
            elif "values" in val:
                raw_data = val["values"]
            
            # 提取元数据判断类型
            if "Unit" in val and isinstance(val["Unit"], dict):
                symbol = val["Unit"].get("Symbol", "").lower() # V, A
                quantity = val["Unit"].get("BaseQuantity", "").lower() # potential, current
                
                if symbol == "v" or "potential" in quantity or "voltage" in quantity:
                    axis_type = "potential"
                elif symbol == "a" or "current" in quantity:
                    axis_type = "current"
            
            # 如果没 Unit，尝试靠 Key 名字猜
            if axis_type == "unknown":
                k_low = key.lower()
                if "xaxis" in k_low or "potential" in k_low: axis_type = "potential"
                elif "yaxis" in k_low or "current" in k_low: axis_type = "current"

        # --- 分支 2: 属性是列表 ---
        elif isinstance(val, list):
            raw_data = val
            k_low = key.lower()
            # 简单的键名匹配
            if k_low in ["x", "xvalues", "potential", "e"]: axis_type = "potential"
            elif k_low in ["y", "yvalues", "current", "i"]: axis_type = "current"

        # --- 数据清洗与存储 ---
        if raw_data and axis_type != "unknown":
            clean_values = extract_values_from_complex_list(raw_data)
            if len(clean_values) > 5: # 忽略太短的数据
                if axis_type == "potential":
                    x_candidates.append(clean_values)
                elif axis_type == "current":
                    y_candidates.append(clean_values)

    # 选择最长的候选数据作为最终结果
    best_x = max(x_candidates, key=len) if x_candidates else []
    best_y = max(y_candidates, key=len) if y_candidates else []
    
    return best_x, best_y

# --- 核心工具 3: 文件解析主入口 ---
def parse_pssession(file):
    datasets = {}
    try:
        # 1. 读取内容
        content = file.getvalue().decode('utf-8', errors='ignore')
        
        # 2. 循环解析 JSON (解决 Extra Data 错误)
        decoder = json.JSONDecoder()
        pos = 0
        all_json_objects = []
        
        while pos < len(content):
            # 跳过空白字符
            while pos < len(content) and content[pos].isspace(): pos += 1
            if pos >= len(content): break
            try:
                obj, end_pos = decoder.raw_decode(content, idx=pos)
                all_json_objects.append(obj)
                pos = end_pos
            except json.JSONDecodeError:
                break # 停止解析
        
        # 3. 提取 Curve 数据
        for root_obj in all_json_objects:
            if not isinstance(root_obj, dict): continue
            
            # 寻找 measurements 节点
            measurements = []
            if "measurements" in root_obj: measurements = root_obj["measurements"]
            elif "Measurements" in root_obj: measurements = root_obj["Measurements"]
            elif "curves" in root_obj or "Curves" in root_obj: measurements = [root_obj]
            
            for m_idx, meas in enumerate(measurements):
                if not isinstance(meas, dict): continue
                
                title = meas.get("title", meas.get("Title", ""))
                curves = meas.get("curves", meas.get("Curves", []))
                
                for c_idx, curve in enumerate(curves):
                    # *** 调用智能识别 ***
                    x, y = smart_find_axis_data(curve)
                    
                    if len(x) > 0 and len(y) > 0:
                        # 裁剪对齐
                        min_len = min(len(x), len(y))
                        x = x[:min_len]
                        y = y[:min_len]
                        
                        # 生成名称
                        fname = file.name.rsplit('.', 1)[0]
                        name = fname
                        # 如果含多个曲线，加后缀
                        if len(measurements) > 1: name += f"-{title}"
                        if len(curves) > 1: name += f"-C{c_idx+1}"
                        
                        datasets[name] = pd.DataFrame({'V': x, 'I': y})
                        
    except Exception as e:
        st.error(f"文件 {file.name} 解析失败: {str(e)}")
        
    return datasets

# --- 核心工具 4: CSV/Excel 解析 ---
def parse_spreadsheet(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, header=None)
    else:
        df = pd.read_excel(file, header=None)
    
    res = {}
    row0 = df.iloc[0].values # 名字行
    
    # 双列遍历
    for i in range(0, df.shape[1], 2):
        if i+1 >= df.shape[1]: break
        
        name = str(row0[i]).strip()
        if name in ['nan', '', 'None']: name = f"Sample_{i//2+1}"
        
        # 名字去重
        base, cnt = name, 1
        while name in res:
            name = f"{base}_{cnt}"
            cnt += 1
            
        # 提取数据 (从第3行开始)
        sub = df.iloc[2:, i:i+2]
        sub.columns = ['V', 'I']
        sub = sub.apply(pd.to_numeric, errors='coerce').dropna()
        
        if not sub.empty: res[name] = sub
    return res

# --- 界面 UI ---
with st.sidebar:
    st.header("1. 数据上传")
    files = st.file_uploader("支持 .pssession, .csv, .xlsx", accept_multiple_files=True)
    
    st.header("2. 绘图设置")
    font_fam = st.selectbox("字体", ["Arial", "Times New Roman", "Helvetica"])
    font_sz = st.slider("字号", 10, 24, 14)
    line_w = st.slider("线宽", 0.5, 4.0, 2.0)
    
    st.subheader("单位调整")
    # 默认针对 pssession (A -> uA)
    current_mult = st.selectbox("电流倍率", 
                               [1, 1e3, 1e6], 
                               index=2, # 默认 10^6
                               format_func=lambda x: "x1 (原始)" if x==1 else ("x10³ (A->mA)" if x==1e3 else "x10⁶ (A->µA)"))
    
    potential_mult = st.selectbox("电位倍率",
                                [1, 1e-3],
                                index=0,
                                format_func=lambda x: "x1 (原始)" if x==1 else "x10⁻³ (mV->V)")

    st.subheader("坐标轴")
    xlabel = st.text_input("X轴", "Potential (V vs. RHE)")
    ylabel = st.text_input("Y轴", "Current (µA)")
    rev_x = st.checkbox("翻转 X 轴", False)
    rev_y = st.checkbox("翻转 Y 轴", False)

st.title("📊 电化学 CV 高级绘图")

# --- 主逻辑 ---
data_pool = {}
if files:
    for f in files:
        f.seek(0) # 重置指针
        fname = f.name.lower()
        
        # 根据后缀分发处理
        d = {}
        if fname.endswith(('.pssession', '.json')):
            d = parse_pssession(f)
        else:
            d = parse_spreadsheet(f)
        
        if not d:
            st.warning(f"⚠️ {f.name}: 未提取到数据。")
        data_pool.update(d)

if data_pool:
    st.header("3. 选择曲线")
    sel = st.multiselect("勾选要绘制的数据", list(data_pool.keys()), default=list(data_pool.keys())[:2])
    
    if sel:
        # 自动分配颜色
        cols = st.columns(min(len(sel), 6))
        palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#DC0000', '#7E6148']
        color_map = {}
        for i, name in enumerate(sel):
            with cols[i % 6]:
                color_map[name] = st.color_picker(name, palette[i % len(palette)])
        
        # 绘图 Matplotlib
        mpl.rcParams['font.family'] = font_fam
        mpl.rcParams['font.size'] = font_sz
        mpl.rcParams['axes.linewidth'] = 1.2
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        
        for name in sel:
            df = data_pool[name]
            
            # 简单的单位应用逻辑
            # 注意: 如果您的 CSV 已经是 uA，这里选 x10^6 会变得非常大。
            # 建议: 上传 pssession 时用默认 x10^6。上传 CSV 时改为 x1。
            x_plot = df['V'] * potential_mult
            y_plot = df['I'] * current_mult
            
            ax.plot(x_plot, y_plot, label=name, color=color_map[name], linewidth=line_w)
            
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        if rev_x: ax.invert_xaxis()
        if rev_y: ax.invert_yaxis()
        
        ax.legend(frameon=False)
        ax.tick_params(top=True, right=True)
        
        st.pyplot(fig)
        
        # 导出功能
        c1, c2 = st.columns(2)
        pdf_buf = io.BytesIO()
        fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')
        c1.download_button("📥 下载 PDF (矢量图)", pdf_buf.getvalue(), "cv_plot.pdf", "application/pdf")
        
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
        c2.download_button("📥 下载 PNG (位图)", png_buf.getvalue(), "cv_plot.png", "image/png")

else:
    st.info("👈 请在左侧上传文件 (支持 CSV/Excel/.pssession)")
