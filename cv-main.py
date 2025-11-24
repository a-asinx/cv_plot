import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import io
import json
import os
from scipy.signal import find_peaks

# --- 页面基础配置 ---
st.set_page_config(
    page_title="CV 科研绘图 (中文专业版)",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# 核心解析模块 (保持不变)
# ==========================================

def extract_values_from_list(data_list):
    """从复杂列表中提取数值"""
    if not data_list or not isinstance(data_list, list) or len(data_list) == 0:
        return []
    first = data_list[0]
    if isinstance(first, (int, float)):
        return data_list
    elif isinstance(first, dict):
        if "V" in first: return [item.get("V", 0) for item in data_list]
        if "y" in first: return [item.get("y", 0) for item in data_list]
        if "v" in first: return [item.get("v", 0) for item in data_list]
    return []

def recursive_search_arrays(obj, context_key=""):
    """深度递归搜索数据数组"""
    found_arrays = []
    if isinstance(obj, dict):
        current_type = "unknown"
        if "Unit" in obj and isinstance(obj["Unit"], dict):
            symbol = obj["Unit"].get("Symbol", "").lower()
            quantity = obj["Unit"].get("BaseQuantity", "").lower()
            if symbol == "v" or "potential" in quantity or "voltage" in quantity:
                current_type = "potential"
            elif symbol == "a" or "current" in quantity:
                current_type = "current"
        
        for k, v in obj.items():
            item_type = current_type
            if item_type == "unknown":
                k_low = k.lower()
                if "x" in k_low or "potential" in k_low: item_type = "potential"
                elif "y" in k_low or "current" in k_low: item_type = "current"
            
            if isinstance(v, (dict, list)):
                found_arrays.extend(recursive_search_arrays(v, context_key=k))
                
            if isinstance(v, list) and len(v) > 5:
                if k in ["m_values", "values", "x", "y", "xValues", "yValues"] or item_type != "unknown":
                    clean_data = extract_values_from_list(v)
                    if len(clean_data) > 5:
                        found_arrays.append({
                            "type": item_type,
                            "data": clean_data,
                            "length": len(clean_data),
                            "key": k
                        })
    return found_arrays

def smart_extract_curve(curve_obj):
    """智能配对 X/Y 轴"""
    candidates = recursive_search_arrays(curve_obj)
    best_x, best_y = [], []
    
    potentials = [c for c in candidates if c['type'] == 'potential']
    currents = [c for c in candidates if c['type'] == 'current']
    
    if potentials: best_x = max(potentials, key=lambda x: x['length'])['data']
    if currents: best_y = max(currents, key=lambda x: x['length'])['data']
    
    if not best_x or not best_y:
        by_length = {}
        for c in candidates:
            l = c['length']
            if l not in by_length: by_length[l] = []
            by_length[l].append(c)
        
        for length in sorted(by_length.keys(), reverse=True):
            group = by_length[length]
            if len(group) >= 2:
                x_cand = next((item for item in group if 'x' in item.get('key', '').lower()), None)
                y_cand = next((item for item in group if 'y' in item.get('key', '').lower()), None)
                if not x_cand and not y_cand: x_cand, y_cand = group[0], group[1]
                elif x_cand and not y_cand: y_cand = next((i for i in group if i is not x_cand), None)
                elif y_cand and not x_cand: x_cand = next((i for i in group if i is not y_cand), None)
                
                if x_cand and y_cand:
                    best_x, best_y = x_cand['data'], y_cand['data']
                    break
    return best_x, best_y

def parse_pssession(file):
    datasets = {}
    try:
        content = file.getvalue().decode('utf-8', errors='ignore')
        decoder = json.JSONDecoder()
        pos = 0
        all_json_objects = []
        while pos < len(content):
            while pos < len(content) and content[pos].isspace(): pos += 1
            if pos >= len(content): break
            try:
                obj, end_pos = decoder.raw_decode(content, idx=pos)
                all_json_objects.append(obj)
                pos = end_pos
            except json.JSONDecodeError: break 
        
        for root_obj in all_json_objects:
            if not isinstance(root_obj, dict): continue
            measurements = []
            if "measurements" in root_obj: measurements = root_obj["measurements"]
            elif "Measurements" in root_obj: measurements = root_obj["Measurements"]
            elif "curves" in root_obj or "Curves" in root_obj: measurements = [root_obj]
            
            for meas in measurements:
                if not isinstance(meas, dict): continue
                title = meas.get("title", meas.get("Title", ""))
                curves = meas.get("curves", meas.get("Curves", []))
                for c_idx, curve in enumerate(curves):
                    x, y = smart_extract_curve(curve)
                    if len(x) > 0 and len(y) > 0:
                        min_len = min(len(x), len(y))
                        fname = file.name.rsplit('.', 1)[0]
                        name = fname 
                        if len(measurements) > 1: name += f"-{title}"
                        if len(curves) > 1: name += f"-C{c_idx+1}"
                        datasets[name] = pd.DataFrame({'V': x[:min_len], 'I': y[:min_len]})
    except Exception: pass
    return datasets

def parse_spreadsheet(file):
    if file.name.endswith('.csv'): df = pd.read_csv(file, header=None)
    else: df = pd.read_excel(file, header=None)
    res = {}
    row0 = df.iloc[0].values
    for i in range(0, df.shape[1], 2):
        if i+1 >= df.shape[1]: break
        name = str(row0[i]).strip()
        if name in ['nan', '', 'None']: name = f"Sample_{i//2+1}"
        base, cnt = name, 1
        while name in res:
            name = f"{base}_{cnt}"
            cnt += 1
        sub = df.iloc[2:, i:i+2]
        sub.columns = ['V', 'I']
        sub = sub.apply(pd.to_numeric, errors='coerce').dropna()
        if not sub.empty: res[name] = sub
    return res

# --- 峰值检测函数 ---
def analyze_peaks(x_data, y_data, prominence):
    """寻找峰值"""
    x = np.array(x_data)
    y = np.array(y_data)
    
    # 寻找极大值 (氧化峰/阳极峰)
    peaks_max_idx, _ = find_peaks(y, prominence=prominence)
    
    # 寻找极小值 (还原峰/阴极峰) - 通过反转 Y 轴寻找
    peaks_min_idx, _ = find_peaks(-y, prominence=prominence)
    
    results = []
    # 整理阳极峰
    for idx in peaks_max_idx:
        results.append({
            "类型": "氧化峰 (Anodic)",
            "电位 (V)": x[idx],
            "电流 (µA)": y[idx]
        })
    # 整理阴极峰
    for idx in peaks_min_idx:
        results.append({
            "类型": "还原峰 (Cathodic)",
            "电位 (V)": x[idx],
            "电流 (µA)": y[idx]
        })
        
    return results

# ==========================================
# UI 与 绘图逻辑
# ==========================================

with st.sidebar:
    st.header("1. 数据导入")
    files = st.file_uploader("上传数据 (.pssession, .csv, .xlsx)", accept_multiple_files=True)
    
    st.header("2. 样式与中文支持")
    
    # --- 字体上传逻辑 ---
    st.markdown("**🛠️ 解决中文乱码：**")
    font_file = st.file_uploader("上传中文字体 (.ttf)", type=["ttf"], help="例如 SimHei.ttf 或 Microsoft YaHei.ttf")
    
    custom_font_name = None
    if font_file:
        try:
            font_path = "custom_font.ttf"
            with open(font_path, "wb") as f:
                f.write(font_file.getbuffer())
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            custom_font_name = font_prop.get_name()
            st.success(f"字体 {custom_font_name} 加载成功！")
        except Exception as e:
            st.error(f"字体加载失败: {e}")

    font_options = ["Arial", "Times New Roman", "Helvetica"]
    if custom_font_name:
        font_options.insert(0, custom_font_name) 
    else:
        system_fonts = [f.name for f in fm.fontManager.ttflist]
        if "SimHei" in system_fonts: font_options.insert(0, "SimHei")
        if "Microsoft YaHei" in system_fonts: font_options.insert(0, "Microsoft YaHei")

    font_fam = st.selectbox("选择字体", font_options, index=0)
    font_sz = st.slider("字号", 10, 28, 16)
    line_w = st.slider("线条粗细", 1.0, 5.0, 2.0)
    
    st.subheader("图例设置")
    separate_legend = st.checkbox("独立图例 (Separate Legend)", False, help="勾选后，图注和曲线图将分开生成，方便排版。")
    
    st.subheader("坐标轴")
    box_style = st.checkbox("全边框 (Box Style)", True)
    tick_dir = st.radio("刻度方向", ["in (内)", "out (外)"], index=0)
    
    st.subheader("Y轴 范围/单位控制")
    mult_mode = st.radio("倍率模式", ["常用预设", "自定义"], horizontal=True)
    if mult_mode == "常用预设":
        current_mult = st.selectbox(
            "选择倍率", 
            [1.0, 1e3, 1e6, 1e-3], 
            index=2, # 默认 1e6
            format_func=lambda x: f"x{x:.0e} (推荐: A → µA)" if x==1e6 else (f"x{x} (原始数据)" if x==1 else f"x{x:.0e}")
        )
    else:
        current_mult = st.number_input("输入自定义倍率", value=1e6, format="%.2e")

    # X轴控制
    potential_mult = st.selectbox("X轴 倍率", [1, 1e-3], index=0, format_func=lambda x: "x1 (V)" if x==1 else "mV → V")
    
    # --- 峰值分析设置 ---
    st.header("3. 峰值分析设置")
    show_peaks_on_plot = st.checkbox("在图中标注峰值", False)
    peak_prominence = st.number_input("峰值灵敏度 (Prominence)", value=1.0, step=0.1, help="数值越小越灵敏。")


st.title("📊 CV 科研绘图工具")
if not font_file:
    st.info("💡 提示：如果中文乱码，请在左侧上传中文字体文件。")

# 1. 数据解析
data_pool = {}
if files:
    for f in files:
        f.seek(0)
        d = parse_pssession(f) if f.name.endswith(('.pssession', '.json')) else parse_spreadsheet(f)
        data_pool.update(d)

if data_pool:
    # 2. 数据选择
    all_keys = list(data_pool.keys())
    sel = st.multiselect("选择曲线", all_keys, default=all_keys[:3] if len(all_keys) > 3 else all_keys)
    
    if sel:
        # --- 图例编辑 ---
        with st.expander("📝 图例名称编辑 (支持 LaTeX)", expanded=True):
            col1, col2 = st.columns(2)
            custom_labels = {}
            for idx, name in enumerate(sel):
                with col1 if idx % 2 == 0 else col2:
                    new_label = st.text_input(f"曲线 {idx+1}", value=name, key=f"lbl_{name}")
                    custom_labels[name] = new_label
        
        c1, c2 = st.columns(2)
        xlabel = c1.text_input("X 轴标签", "Potential (V)")
        ylabel = c2.text_input("Y 轴标签", "Current (μA)") 

        cols = st.columns(len(sel))
        palette = ['#CC3333', '#3366CC', '#009966', '#FF9900', '#9933CC', '#666666']
        color_map = {}
        for i, name in enumerate(sel):
            with cols[i % len(cols)]:
                color_map[name] = st.color_picker(f"Color: {custom_labels[name]}", palette[i % len(palette)])

        # --- 绘图配置 ---
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['axes.unicode_minus'] = False 
        
        if custom_font_name:
            mpl.rcParams['font.sans-serif'] = [custom_font_name, 'Arial']
        else:
            mpl.rcParams['font.sans-serif'] = [font_fam, 'SimHei', 'Arial']
            
        mpl.rcParams['font.size'] = font_sz
        mpl.rcParams['axes.linewidth'] = 1.5
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(6, 4.8), dpi=150)
        
        all_peak_data = [] # 存储所有峰值数据用于表格
        plot_handles = []  # 存储用于图例的句柄
        plot_labels = []   # 存储用于图例的标签

        for name in sel:
            df = data_pool[name]
            x = df['V'] * potential_mult
            y = df['I'] * current_mult 
            
            label_text = custom_labels[name]
            line, = ax.plot(x, y, label=label_text, color=color_map[name], linewidth=line_w)
            
            # 收集句柄用于独立图例
            plot_handles.append(line)
            plot_labels.append(label_text)
            
            # --- 峰值计算 ---
            peaks = analyze_peaks(x, y, peak_prominence)
            if peaks:
                for p in peaks:
                    p["曲线名称"] = label_text
                    all_peak_data.append(p)
                    if show_peaks_on_plot:
                        ax.plot(p["电位 (V)"], p["电流 (µA)"], "x", color='black', markersize=6)
        
        # 样式复刻
        ax.set_xlabel(xlabel, fontweight='bold', labelpad=10)
        ax.set_ylabel(ylabel, fontweight='bold', labelpad=10)
        
        tick_direction = 'in' if 'in' in tick_dir else 'out'
        ax.tick_params(which='major', direction=tick_direction, length=6, width=1.5, 
                       top=box_style, right=box_style, bottom=True, left=True)
        ax.minorticks_on()
        ax.tick_params(which='minor', direction=tick_direction, length=3, width=1.0, 
                       top=box_style, right=box_style, bottom=True, left=True)

        if box_style:
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
        
        # --- 图例处理 ---
        if not separate_legend:
            # 默认模式：图例在图内
            ax.legend(frameon=False, fontsize=font_sz-2, loc='best')
        else:
            # 独立图例模式：不在主图绘制图例
            pass

        plt.tight_layout()

        # === 布局展示 ===
        if not separate_legend:
            # 模式 A: 正常显示
            st.pyplot(fig, use_container_width=True)
            
            # 导出图片
            col1, col2 = st.columns(2)
            pdf_buf = io.BytesIO()
            fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')
            col1.download_button("📥 下载完整 PDF", pdf_buf.getvalue(), "cv_plot.pdf", "application/pdf")
            
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
            col2.download_button("📥 下载完整 PNG", png_buf.getvalue(), "cv_plot.png", "image/png")
            
        else:
            # 模式 B: 分离显示
            st.markdown("### 预览 (分离模式)")
            c_plot, c_legend = st.columns([3, 1])
            
            # 1. 展示主图 (无图例)
            with c_plot:
                st.markdown("**1. 纯净曲线图**")
                st.pyplot(fig, use_container_width=True)
                
                # 导出主图
                pdf_plot = io.BytesIO()
                fig.savefig(pdf_plot, format='pdf', bbox_inches='tight')
                st.download_button("📥 下载曲线图 PDF", pdf_plot.getvalue(), "cv_curve_only.pdf", "application/pdf")
                
                png_plot = io.BytesIO()
                fig.savefig(png_plot, format='png', dpi=300, bbox_inches='tight')
                st.download_button("📥 下载曲线图 PNG", png_plot.getvalue(), "cv_curve_only.png", "image/png")

            # 2. 生成并展示独立图例
            with c_legend:
                st.markdown("**2. 独立图例**")
                # 创建专门画图例的 figure
                fig_leg, ax_leg = plt.subplots(figsize=(2, 0.5 * len(sel) + 0.5), dpi=150)
                ax_leg.axis('off') # 隐藏坐标轴
                ax_leg.legend(plot_handles, plot_labels, loc='center', frameon=False, fontsize=font_sz)
                st.pyplot(fig_leg, use_container_width=True)
                
                # 导出图例
                pdf_leg = io.BytesIO()
                fig_leg.savefig(pdf_leg, format='pdf', bbox_inches='tight')
                st.download_button("📥 下载图例 PDF", pdf_leg.getvalue(), "cv_legend_only.pdf", "application/pdf")
                
                png_leg = io.BytesIO()
                fig_leg.savefig(png_leg, format='png', dpi=300, bbox_inches='tight')
                st.download_button("📥 下载图例 PNG", png_leg.getvalue(), "cv_legend_only.png", "image/png")
        
        # --- 4. 峰值数据展示区 ---
        st.markdown("---")
        st.markdown("### 4. 峰值数据统计")
        if all_peak_data:
            peak_df = pd.DataFrame(all_peak_data)
            cols_order = ["曲线名称", "类型", "电位 (V)", "电流 (µA)"]
            peak_df = peak_df[cols_order]
            st.dataframe(peak_df, use_container_width=True)
            csv = peak_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 导出峰值数据 (CSV)", csv, "cv_peaks.csv", "text/csv")
        else:
            st.info("未检测到明显的峰值。请调整“峰值灵敏度”。")

else:
    st.info("👈 请在左侧上传数据文件开始。")
