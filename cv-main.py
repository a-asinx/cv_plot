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
    修复 'Extra data' 错误：使用 raw_decode 循环读取有效 JSON 块，忽略末尾垃圾数据
    """
    datasets = {}
    try:
        # 1. 获取文件内容并解码为字符串 (忽略非 UTF-8 的二进制尾部)
        content = file.getvalue().decode('utf-8', errors='ignore')
        
        # 2. 循环解析所有 JSON 对象
        decoder = json.JSONDecoder()
        pos = 0
        all_json_objects = []
        
        while pos < len(content):
            # 跳过空白字符
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            
            try:
                # raw_decode 会返回解析出的对象和结束位置的索引
                obj, end_pos = decoder.raw_decode(content, idx=pos)
                all_json_objects.append(obj)
                pos = end_pos
            except json.JSONDecodeError:
                # 如果遇到无法解析的部分（比如文件末尾的非JSON数据），直接停止，保留已解析的部分
                break
        
        # 3. 在所有解析出的对象中寻找 measurement 数据
        for data_json in all_json_objects:
            measurements = []
            
            # 尝试不同的键名结构
            if isinstance(data_json, dict):
                if "measurements" in data_json:
                    measurements = data_json["measurements"]
                elif "Measurements" in data_json:
                    measurements = data_json["Measurements"]
                elif "curves" in data_json or "Curves" in data_json:
                    # 有些对象直接就是 measurement 本身
                    measurements = [data_json]
            
            # 遍历 measurement 提取曲线
            for m_idx, meas in enumerate(measurements):
                if not isinstance(meas, dict): continue
                
                title = meas.get("title", meas.get("Title", f"Meas"))
                
                # 获取曲线列表
                curves = meas.get("curves", meas.get("Curves", []))
                
                for c_idx, curve in enumerate(curves):
                    # 尝试获取 x 和 y 数组
                    # PalmSens 常见键名: x, xValues, X, y, yValues, Y
                    x = curve.get("x", curve.get("xValues", curve.get("X", [])))
                    y = curve.get("y", curve.get("yValues", curve.get("Y", [])))
                    
                    if x and y and len(x) > 0 and len(y) > 0:
                        # 构建名称
                        # 使用文件名作为前缀，避免多文件混淆
                        clean_fname = file.name.rsplit('.', 1)[0]
                        name = f"{clean_fname}"
                        
                        # 只有当文件里包含多个 measurement 时才加后缀，保持图例简洁
                        if len(all_json_objects) > 1 or len(measurements) > 1:
                            name += f"_{title}"
                        if len(curves) > 1:
                            name += f"_Curve{c_idx+1}"
                        
                        # 存入 DataFrame
                        df = pd.DataFrame({'V': x, 'I': y})
                        datasets[name] = df
                    
    except Exception as e:
        st.error(f"解析 .pssession 文件 {file.name} 时出错: {str(e)}")
        
    return datasets

# --- 核心逻辑：解析 CSV/Excel ---
def parse_spreadsheet(file):
    """
    解析双行表头格式：
    Row 0: Sample Name, Empty, Sample Name 2...
    Row 1: V, I, V, I...
    """
    filename = file.name
    if filename.endswith('.csv'):
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
                               index=2, # 默认选中 x10^6 (A->uA) 因为 pssession 通常是 A
                               format_func=lambda x: "x1 (原始)" if x==1 else ("x10³ (A->mA)" if x==1e3 else "x10⁶ (A->µA)"))
    
    # 坐标轴
    st.subheader("坐标轴")
    x_label = st.text_input("X 轴标签", "Potential (V vs. RHE)")
    y_label = st.text_input("Y 轴标签", "Current (µA)")
    reverse_x = st.checkbox("翻转 X 轴 (Reverse Scan)", value=False)
    reverse_y = st.checkbox("翻转 Y 轴 (IUPAC vs US)", value=False)

# 处理数据
all_datasets = {}
if uploaded_files:
    for f in uploaded_files:
        # 将指针重置，以防多次读取
        f.seek(0)
        fname = f.name.lower()
        
        # 智能判断解析方式
        if fname.endswith(('.pssession', '.json')):
            d = parse_pssession(f)
        else:
            d = parse_spreadsheet(f)
            
        if not d:
            st.warning(f"文件 {f.name} 中未找到有效数据，请检查格式。")
            
        all_datasets.update(d)

# 显示选择区域和图表
if all_datasets:
    st.header("数据选择")
    selected_names = st.multiselect("选择要对比的曲线", list(all_datasets.keys()), default=list(all_datasets.keys())[:2])
    
    if selected_names:
        # 颜色配置
        cols = st.columns(len(selected_names) if len(selected_names)<5 else 5)
        colors = {}
        default_palette = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
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
            
            # 判断是否需要应用乘数
            # 简单的启发式规则：如果文件名看起来像 CSV，可能已经是 uA 了，不需要再乘
            # 但为了简单，这里统一受侧边栏控制。
            # 如果 CSV 数据很大（已经是 uA），用户选 x1 即可。
            # .pssession 数据通常很小（A），默认选 x10^6 即可。
            
            y_data = df['I'] * current_mult
            
            ax.plot(df['V'], y_data, label=name, color=colors[name], linewidth=line_width)
            
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        
        if reverse_x: ax.invert_xaxis()
        if reverse_y: ax.invert_yaxis()
            
        # 高水平期刊风格
        ax.legend(frameon=False)
        ax.tick_params(top=True, right=True)
        
        st.pyplot(fig)
        
        # 导出
        st.subheader("导出图片")
        col1, col2 = st.columns(2)
        # PDF
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight')
        col1.download_button("📥 下载 PDF (矢量图)", pdf_buffer.getvalue(), "cv_plot.pdf", "application/pdf")
        # PNG
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
        col2.download_button("📥 下载 PNG (高清位图)", png_buffer.getvalue(), "cv_plot.png", "image/png")
        
    else:
        st.info("请至少选择一条曲线。")
else:
    st.info("👈 请在左侧上传您的 CSV, Excel 或 .pssession 文件。")
