import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import io
import json

# --- 页面基础配置 ---
st.set_page_config(
    page_title="CV 数据分析与绘图 (全能版)",
    page_icon="⚡",
    layout="wide"
)

# --- 核心工具 1: 复杂列表数值提取 ---
def extract_values_from_list(data_list):
    """
    从 PalmSens 复杂的列表结构中提取数值。
    支持:
    1. [1.1, 1.2, ...] (纯数字)
    2. [{"V": 1.1}, {"V": 1.2}, ...] (对象包装, V=Value)
    """
    if not data_list or not isinstance(data_list, list) or len(data_list) == 0:
        return []

    first = data_list[0]
    
    # 情况 A: 纯数字
    if isinstance(first, (int, float)):
        return data_list
    
    # 情况 B: 字典对象 ( PalmSens 序列化格式 )
    elif isinstance(first, dict):
        # 这里的 V 代表 Value (数值), 不一定是 Voltage
        if "V" in first: return [item.get("V", 0) for item in data_list]
        if "y" in first: return [item.get("y", 0) for item in data_list]
        if "v" in first: return [item.get("v", 0) for item in data_list]
            
    return []

# --- 核心工具 2: 递归搜索数据 ---
def recursive_search_arrays(obj, context_key=""):
    """
    深度递归搜索：遍历 JSON 树的每一个节点，收集所有可能是数据的数组。
    返回: List of dicts [{'type': 'potential'/'current'/'unknown', 'data': [], 'score': int}]
    """
    found_arrays = []

    if isinstance(obj, dict):
        # --- 1. 检查当前对象是否包含元数据 (Unit) ---
        current_type = "unknown"
        if "Unit" in obj and isinstance(obj["Unit"], dict):
            symbol = obj["Unit"].get("Symbol", "").lower()
            quantity = obj["Unit"].get("BaseQuantity", "").lower()
            
            if symbol == "v" or "potential" in quantity or "voltage" in quantity:
                current_type = "potential"
            elif symbol == "a" or "current" in quantity:
                current_type = "current"
        
        # --- 2. 遍历字典的键值 ---
        for k, v in obj.items():
            # 结合父级上下文推断类型 (如果元数据没找到)
            item_type = current_type
            if item_type == "unknown":
                k_low = k.lower()
                # 根据键名猜测
                if "x" in k_low or "potential" in k_low: item_type = "potential"
                elif "y" in k_low or "current" in k_low: item_type = "current"
            
            # 递归下钻
            if isinstance(v, (dict, list)):
                found_arrays.extend(recursive_search_arrays(v, context_key=k))
                
            # --- 3. 检查当前值是否为目标数组 ---
            # 特征：必须是列表，且长度大于5，且键名看起来像数据
            if isinstance(v, list) and len(v) > 5:
                # 进一步验证内容是否为数字
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
    """
    使用递归搜索结果，智能配对 X 和 Y 轴。
    """
    # 1. 全局搜索该 Curve 对象下的所有数组
    candidates = recursive_search_arrays(curve_obj)
    
    best_x = []
    best_y = []
    
    # 2. 筛选策略
    # 优先找明确标记为 Potential 和 Current 的最长数组
    potentials = [c for c in candidates if c['type'] == 'potential']
    currents = [c for c in candidates if c['type'] == 'current']
    
    if potentials: best_x = max(potentials, key=lambda x: x['length'])['data']
    if currents: best_y = max(currents, key=lambda x: x['length'])['data']
    
    # 3. 补救策略 (如果没有明确的 Unit 标记)
    # 如果找不到明确类型，但找到了两个长度一致的长数组，尝试按键名或顺序猜测
    if not best_x or not best_y:
        # 按长度分组
        by_length = {}
        for c in candidates:
            l = c['length']
            if l not in by_length: by_length[l] = []
            by_length[l].append(c)
        
        # 找包含至少两个数组的最长长度组
        for length in sorted(by_length.keys(), reverse=True):
            group = by_length[length]
            if len(group) >= 2:
                # 这一组里大概率一个是X一个是Y
                # 尝试找 X 候选
                x_cand = next((item for item in group if 'x' in item.get('key', '').lower()), None)
                y_cand = next((item for item in group if 'y' in item.get('key', '').lower()), None)
                
                # 如果没名字特征，默认第一个是X(PalmSens常见顺序)? 不，这有风险。
                # 但通常 key="xValues" 或 key="m_values" (在XAxisDataArray下)
                if not x_cand and not y_cand:
                    x_cand = group[0]
                    y_cand = group[1]
                elif x_cand and not y_cand:
                    # 剩下的那个是 Y
                    y_cand = next((item for item in group if item is not x_cand), None)
                elif y_cand and not x_cand:
                    x_cand = next((item for item in group if item is not y_cand), None)
                
                if x_cand and y_cand:
                    best_x = x_cand['data']
                    best_y = y_cand['data']
                    break
                    
    return best_x, best_y

# --- 核心工具 3: 文件解析主入口 ---
def parse_pssession(file):
    datasets = {}
    try:
        # 1. 鲁棒读取 (Raw Decode 循环)
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
            except json.JSONDecodeError:
                break 
        
        # 2. 遍历提取
        for root_obj in all_json_objects:
            if not isinstance(root_obj, dict): continue
            
            # 定位 Measurements
            measurements = []
            if "measurements" in root_obj: measurements = root_obj["measurements"]
            elif "Measurements" in root_obj: measurements = root_obj["Measurements"]
            elif "curves" in root_obj or "Curves" in root_obj: measurements = [root_obj]
            
            for m_idx, meas in enumerate(measurements):
                if not isinstance(meas, dict): continue
                title = meas.get("title", meas.get("Title", ""))
                curves = meas.get("curves", meas.get("Curves", []))
                
                for c_idx, curve in enumerate(curves):
                    # *** 调用深度递归提取 ***
                    x, y = smart_extract_curve(curve)
                    
                    if len(x) > 0 and len(y) > 0:
                        min_len = min(len(x), len(y))
                        x = x[:min_len]
                        y = y[:min_len]
                        
                        fname = file.name.rsplit('.', 1)[0]
                        name = fname
                        if len(measurements) > 1: name += f"-{title}"
                        if len(curves) > 1: name += f"-C{c_idx+1}"
                        
                        datasets[name] = pd.DataFrame({'V': x, 'I': y})
                        
    except Exception as e:
        st.error(f"文件 {file.name} 解析严重错误: {str(e)}")
        
    return datasets

# --- 核心工具 4: CSV/Excel 解析 ---
def parse_spreadsheet(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, header=None)
    else:
        df = pd.read_excel(file, header=None)
    
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

# --- 界面 UI ---
with st.sidebar:
    st.header("1. 数据上传")
    files = st.file_uploader("支持 .pssession, .csv, .xlsx", accept_multiple_files=True)
    
    st.header("2. 绘图设置")
    font_fam = st.selectbox("字体", ["Arial", "Times New Roman", "Helvetica"])
    font_sz = st.slider("字号", 10, 24, 14)
    line_w = st.slider("线宽", 0.5, 4.0, 2.0)
    
    st.subheader("单位调整")
    # pssession 默认 A -> uA (x1e6)
    current_mult = st.selectbox("电流倍率", [1, 1e3, 1e6], index=2,
                               format_func=lambda x: "x1 (原始)" if x==1 else ("x10³ (mA)" if x==1e3 else "x10⁶ (µA)"))
    potential_mult = st.selectbox("电位倍率", [1, 1e-3], index=0,
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
        f.seek(0)
        fname = f.name.lower()
        d = {}
        if fname.endswith(('.pssession', '.json')):
            d = parse_pssession(f)
        else:
            d = parse_spreadsheet(f)
        
        if not d:
            st.warning(f"⚠️ {f.name}: 未提取到数据。请确认文件是否包含有效测量数据。")
        data_pool.update(d)

if data_pool:
    st.header("3. 选择曲线")
    sel = st.multiselect("勾选要绘制的数据", list(data_pool.keys()), default=list(data_pool.keys())[:2])
    
    if sel:
        cols = st.columns(min(len(sel), 6))
        palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#DC0000', '#7E6148']
        color_map = {}
        for i, name in enumerate(sel):
            with cols[i % 6]:
                color_map[name] = st.color_picker(name, palette[i % len(palette)])
        
        mpl.rcParams['font.family'] = font_fam
        mpl.rcParams['font.size'] = font_sz
        mpl.rcParams['axes.linewidth'] = 1.2
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        
        for name in sel:
            df = data_pool[name]
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
        
        c1, c2 = st.columns(2)
        pdf_buf = io.BytesIO()
        fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')
        c1.download_button("📥 下载 PDF", pdf_buf.getvalue(), "cv_plot.pdf", "application/pdf")
        
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
        c2.download_button("📥 下载 PNG", png_buf.getvalue(), "cv_plot.png", "image/png")

else:
    st.info("👈 请在左侧上传文件 (支持 CSV/Excel/.pssession)")
