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

# --- 核心逻辑：解析 .pssession (JSON) ---
def parse_pssession(file):
    """
    解析 PalmSens .pssession (JSON格式) 文件
    尝试从 measurements -> curves 中提取 x 和 y 数据
    """
    datasets = {}
    try:
        content = file.read()
        data_json = json.loads(content)
        
        # 兼容不同版本的 JSON 结构
        measurements = []
        if "measurements" in data_json:
            measurements = data_json["measurements"]
        elif "Measurements" in data_json:
            measurements = data_json["Measurements"]
        # 有些文件根节点就是单个 measurement
        elif "curves" in data_json or "Curves" in data_json:
            measurements = [data_json]

        for m_idx, meas in enumerate(measurements):
            title = meas.get("title", meas.get("Title", f"Meas_{m_idx+1}"))
            
            # 获取曲线列表
            curves = meas.get("curves", meas.get("Curves", []))
            
            for c_idx, curve in enumerate(curves):
                # 尝试获取 x 和 y 数组
                # PalmSens 常见键名: x, xValues, X, y, yValues, Y
                x = curve.get("x", curve.get("xValues", curve.get("X", [])))
                y = curve.get("y", curve.get("yValues", curve.get("Y", [])))
                
                if len(x) > 0 and len(y) > 0:
                    # 构建名称
                    name = f"{file.name.split('.')[0]}"
                    if len(measurements) > 1:
                        name += f"_{title}"
                    if len(curves) > 1:
                        name += f"_Curve{c_idx+1}"
                    
                    df = pd.DataFrame({'V': x, 'I': y})
                    datasets[name] = df
                    
    except Exception as e:
        st.error(f"解析 .pssession 出错: {e}")
        
    return datasets

# --- 核心逻辑：解析 CSV/Excel ---
def parse_spreadsheet(file):
    """
    解析双行表头格式：
    Row 0: Sample Name, Empty, Sample Name 2...
    Row 1: V, I, V, I...
    """
    if file.name.endswith('.csv'):
        df_raw = pd.read_csv(file, header=None)
    else:
        df_raw = pd.read_excel(file, header=None)

    datasets = {}
    row0 = df_raw.iloc[0].values
    
    # 遍历列，步长为2
    for i in range(0, df_raw.shape[1], 2):
        if i + 1 >= df_raw.shape[1]: break
        
        # 获取样品名
        name = str(row0[i]).strip()
        if name in ['nan', '', 'None']: 
            name = f"Sample_{i//2 + 1}"
            
        # 处理重名
        base_name = name
        counter = 1
        while name in datasets:
            name = f"{base_name}_{counter}"
            counter += 1
            
        # 提取数据 (跳过前两行表头)
        sub_df = df_raw.iloc[2:, i:i+2]
        sub_df.columns = ['V', 'I']
        sub_df = sub_df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if not sub_df.empty:
            datasets[name] = sub_df
            
    return datasets

# --- 主界面逻辑 ---
st.title("🔬 电化学 CV 数据对比与绘图")
st.markdown("支持格式：**CSV / Excel** (双行表头) 以及 **.pssession** (PalmSens JSON)")

# 侧边栏：控制面板
with st.sidebar:
    st.header("1. 数据上传")
    uploaded_files = st.file_uploader("选择文件", type=['csv', 'xlsx', 'xls', 'pssession', 'json'], accept_multiple_files=True)
    
    st.header("2. 绘图设置")
    # 样式设置
    font_family = st.selectbox("字体", ["Arial", "Times New Roman", "Helvetica"], index=0)
    font_size = st.slider("字号", 10, 24, 14)
    line_width = st.slider("线宽", 0.5, 4.0, 2.0)
    
    # 单位处理
    st.subheader("单位转换")
    current_mult = st.selectbox("电流乘数 (用于 .pssession)", 
                               [1, 1e3, 1e6], 
                               format_func=lambda x: "x1 (原始)" if x==1 else ("x10³ (A->mA)" if x==1e3 else "x10⁶ (A->µA)"))
    
    # 坐标轴
    st.subheader("坐标轴")
    x_label = st.text_input("X 轴标签", "Potential (V vs. RHE)")
    y_label = st.text_input("Y 轴标签", "Current (µA)")
    reverse_x = st.checkbox("翻转 X 轴 (Reverse Scan)", value=False)

# 处理数据
all_datasets = {}
if uploaded_files:
    for f in uploaded_files:
        # 将指针重置，以防多次读取
        f.seek(0)
        if f.name.endswith(('.pssession', '.json')):
            d = parse_pssession(f)
        else:
            d = parse_spreadsheet(f)
        all_datasets.update(d)

# 显示选择区域和图表
if all_datasets:
    st.header("数据选择")
    selected_names = st.multiselect("选择要对比的曲线", list(all_datasets.keys()), default=list(all_datasets.keys())[:2])
    
    if selected_names:
        # 颜色配置
        cols = st.columns(len(selected_names) if len(selected_names)<5 else 5)
        colors = {}
        default_palette = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        for idx, name in enumerate(selected_names):
            with cols[idx % 5]:
                colors[name] = st.color_picker(name, default_palette[idx % len(default_palette)])
        
        # 绘图
        mpl.rcParams['font.family'] = font_family
        mpl.rcParams['font.size'] = font_size
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
        
        for name in selected_names:
            df = all_datasets[name]
            # 应用电流乘数 (主要针对 pssession 的 A -> uA)
            # 如果是 CSV，通常已经是 uA 了，所以只对 pssession 来源应用可能更合理
            # 这里为了简单，全局应用。如果 CSV 已经是 uA，选 x1 即可。
            y_data = df['I'] * current_mult
            
            ax.plot(df['V'], y_data, label=name, color=colors[name], linewidth=line_width)
            
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        
        if reverse_x:
            ax.invert_xaxis()
            
        # 高水平期刊风格：图例无框，刻度向内
        ax.legend(frameon=False)
        ax.tick_params(top=True, right=True)
        
        st.pyplot(fig)
        
        # 导出
        st.subheader("导出图片")
        col1, col2 = st.columns(2)
        # PDF
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight')
        col1.download_button("下载 PDF (矢量图)", pdf_buffer.getvalue(), "cv_plot.pdf", "application/pdf")
        # PNG
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
        col2.download_button("下载 PNG (高清位图)", png_buffer.getvalue(), "cv_plot.png", "image/png")
        
    else:
        st.info("请至少选择一条曲线。")
else:
    st.info("👈 请在左侧上传您的 CSV, Excel 或 .pssession 文件。")
