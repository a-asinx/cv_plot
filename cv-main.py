import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import io

# --- 页面配置 ---
st.set_page_config(
    page_title="CV 电化学数据可视化工具",
    page_icon="⚡",
    layout="wide"
)


# --- 工具函数：解析复杂格式的 CSV/Excel ---
def parse_cv_data(file):
    """
    解析特定的 CV 数据格式：
    Row 0: 样品名称 (Sample Name), Empty, Sample Name 2, Empty...
    Row 1: V, µA, V, µA...
    Row 2+: Data
    """
    # 判断文件类型
    filename = file.name
    if filename.endswith('.csv'):
        df_raw = pd.read_csv(file, header=None)
    else:
        df_raw = pd.read_excel(file, header=None)

    # 提取第一行作为名称，填充 NaN (处理合并单元格产生的空值)
    # 假设每两列是一组数据
    datasets = {}

    # 获取第一行数据
    row0 = df_raw.iloc[0].values

    # 遍历列，步长为2 (V 和 Current)
    num_cols = df_raw.shape[1]
    for i in range(0, num_cols, 2):
        if i + 1 >= num_cols:
            break

        sample_name = str(row0[i]).strip()
        if sample_name == 'nan' or sample_name == '':
            sample_name = f"Sample_{i // 2 + 1}"

        # 为了防止重名，如果名字已存在，加后缀
        original_name = sample_name
        counter = 1
        while sample_name in datasets:
            sample_name = f"{original_name}_{counter}"
            counter += 1

        # 提取数据 (从第2行开始是表头单位，第3行开始是数值，但这里我们直接取数值部分)
        # 假设第2行(index 1)是单位 V, A，第3行(index 2)开始是数据
        sub_df = df_raw.iloc[2:, i:i + 2]
        sub_df.columns = ['V', 'I']

        # 强制转换为数值，去除可能的非法字符
        sub_df['V'] = pd.to_numeric(sub_df['V'], errors='coerce')
        sub_df['I'] = pd.to_numeric(sub_df['I'], errors='coerce')

        # 删除空行
        sub_df = sub_df.dropna()

        if not sub_df.empty:
            datasets[sample_name] = sub_df

    return datasets


# --- 主界面 ---
st.title("⚡ 电化学循环伏安 (CV) 高级绘图工具")
st.markdown("""
上传您的 CV 测试数据文件 (CSV 或 Excel)，选择特定的圈数/样品进行对比，并导出符合 **SCI 期刊标准** 的高清图片。
""")

# --- 侧边栏：全局设置 ---
st.sidebar.header("1. 上传数据")
uploaded_file = st.sidebar.file_uploader("上传文件 (CSV/XLSX)", type=["csv", "xlsx", "xls"])

st.sidebar.header("3. 绘图参数设置")
st.sidebar.subheader("通用样式")
font_family = st.sidebar.selectbox("字体 (Font Family)", ["Arial", "Times New Roman", "Helvetica", "DejaVu Sans"],
                                   index=0)
font_size = st.sidebar.slider("基础字号 (Font Size)", 8, 24, 12)
line_width = st.sidebar.slider("线条宽度 (Line Width)", 0.5, 5.0, 1.5)
fig_width = st.sidebar.slider("图片宽度 (inch)", 3.0, 12.0, 6.0)
fig_height = st.sidebar.slider("图片高度 (inch)", 3.0, 10.0, 4.5)

st.sidebar.subheader("坐标轴设置")
x_label = st.sidebar.text_input("X 轴标签", "Potential (V vs. RHE)")
y_label = st.sidebar.text_input("Y 轴标签", "Current (µA)")
x_tick_dir = st.sidebar.radio("X 轴刻度方向", ["in", "out"], index=0)
y_tick_dir = st.sidebar.radio("Y 轴刻度方向", ["in", "out"], index=0)
show_grid = st.sidebar.checkbox("显示网格 (Grid)", False)

st.sidebar.subheader("图例设置")
show_legend = st.sidebar.checkbox("显示图例 (Legend)", True)
legend_loc = st.sidebar.selectbox("图例位置", ["best", "upper right", "upper left", "lower right", "lower left"],
                                  index=0)
frame_on = st.sidebar.checkbox("图例边框", False)

# --- 主逻辑 ---
if uploaded_file is not None:
    try:
        # 1. 解析数据
        datasets = parse_cv_data(uploaded_file)
        st.success(f"成功读取文件，共识别出 {len(datasets)} 组数据。")

        # 2. 数据选择区域
        st.header("2. 选择数据进行对比")
        all_keys = list(datasets.keys())

        # 默认全选，如果太多则默认选前两个
        default_selection = all_keys[:2] if len(all_keys) > 0 else []
        selected_keys = st.multiselect("请选择要绘制的曲线：", all_keys, default=default_selection)

        if selected_keys:
            # 3. 颜色映射
            st.markdown("#### 🎨 颜色自定义")
            cols = st.columns(4)
            color_map = {}
            # 预定义一些好看的学术配色
            default_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf']

            for idx, key in enumerate(selected_keys):
                with cols[idx % 4]:
                    default_c = default_colors[idx % len(default_colors)]
                    color_map[key] = st.color_picker(f"{key}", default_c)

            # 4. 绘图逻辑 (Matplotlib)
            # 设置全局字体
            mpl.rcParams['font.family'] = font_family
            mpl.rcParams['font.size'] = font_size
            mpl.rcParams['axes.linewidth'] = 1.0  # 边框粗细

            # 创建画布
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

            for key in selected_keys:
                data = datasets[key]
                ax.plot(data['V'], data['I'],
                        label=key,
                        linewidth=line_width,
                        color=color_map[key])

            # 轴标签
            ax.set_xlabel(x_label, fontweight='bold')
            ax.set_ylabel(y_label, fontweight='bold')

            # 刻度设置 (高水平期刊通常要求刻度朝内)
            ax.tick_params(direction=x_tick_dir, length=6, width=1, which='major', top=True, right=True)
            ax.tick_params(direction=x_tick_dir, length=3, width=1, which='minor', top=True, right=True)

            # 网格
            if show_grid:
                ax.grid(True, linestyle='--', alpha=0.6)

            # 图例
            if show_legend:
                ax.legend(loc=legend_loc, frameon=frame_on, fontsize=font_size - 2)

            # 布局调整
            plt.tight_layout()

            # 5. 展示图片
            st.pyplot(fig)

            # 6. 导出功能
            st.header("4. 导出图片")
            col1, col2 = st.columns(2)

            # 保存为 PNG
            img_buffer_png = io.BytesIO()
            fig.savefig(img_buffer_png, format='png', dpi=300, bbox_inches='tight')
            img_buffer_png.seek(0)
            col1.download_button(
                label="📥 下载高分辨率 PNG (300 DPI)",
                data=img_buffer_png,
                file_name="cv_plot_high_res.png",
                mime="image/png"
            )

            # 保存为 PDF (矢量图，最佳用于插入论文)
            img_buffer_pdf = io.BytesIO()
            fig.savefig(img_buffer_pdf, format='pdf', bbox_inches='tight')
            img_buffer_pdf.seek(0)
            col2.download_button(
                label="📥 下载 PDF (矢量图/期刊推荐)",
                data=img_buffer_pdf,
                file_name="cv_plot_vector.pdf",
                mime="application/pdf"
            )

        else:
            st.info("请在上方选择至少一条曲线进行绘制。")

    except Exception as e:
        st.error(f"处理文件时出错: {e}")
        st.warning("请确保文件格式为：第一行是样品名，第二行是单位(V, A)，数据为成对列排列。")