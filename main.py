import streamlit as st
import pandas as pd
from dataset import load_data, write_data, CATEGORY_LIST
from augmentation_total import augment_dataframe, save_augmented_data
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="3D Motion Data Visualization & Augmentation", layout="wide")

#데이터 로드 & 증강 설정
st.sidebar.title("Settings")

# 데이터 로드
st.sidebar.header("Load Data")
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["Original (data1 + data2)", "data1 only (clean)", "data2 only (noisy)"]
)

if data_source == "Original (data1 + data2)":
    data1 = load_data("data1")
    data2 = load_data("data2")
    total_data = pd.concat([data1, data2])
    st.sidebar.success(f"Loaded: data1 + data2")
elif data_source == "data1 only (clean)":
    total_data = load_data("data1")
    st.sidebar.success(f"Loaded: data1 (clean)")
else:
    total_data = load_data("data2")
    st.sidebar.success(f"Loaded: data2 (noisy)")

original_data = total_data.copy()

# 세션 상태 초기화
if 'augmented_data' not in st.session_state:
    st.session_state.augmented_data = None

# 증강 기능
st.sidebar.header("Data Augmentation")
enable_augmentation = st.sidebar.checkbox("Enable Data Augmentation", value=False)

if enable_augmentation:
    st.sidebar.subheader("Select Augmentation Methods")

    # 증강 기법 선택
    use_time_warp = st.sidebar.checkbox("Time Warping", value=True)
    use_jitter = st.sidebar.checkbox("Add Jitter (Noise)", value=True)
    use_rotate = st.sidebar.checkbox("3D Rotation", value=True)
    use_scale = st.sidebar.checkbox("Scaling", value=True)
    use_shift = st.sidebar.checkbox("Time Shifting", value=False)

    # 증강 파라미터 설정
    st.sidebar.subheader("Augmentation Parameters")
    augment_per_method = st.sidebar.slider("Augmentations per method", 1, 3, 1)

    methods = []
    if use_time_warp:
        warp_factor = st.sidebar.slider("Time Warp Factor", 0.5, 2.0, 1.3, 0.1)
        methods.append({'method': 'time_warp', 'warp_factor': warp_factor})

    if use_jitter:
        sigma = st.sidebar.slider("Jitter Sigma", 0.01, 0.2, 0.05, 0.01)
        methods.append({'method': 'jitter', 'sigma': sigma})

    if use_rotate:
        angle_deg = st.sidebar.slider("Rotation Angle (deg)", -45, 45, 15, 5)
        axis = st.sidebar.selectbox("Rotation Axis", ['x', 'y', 'z'], index=2)
        methods.append({'method': 'rotate', 'angle_deg': angle_deg, 'axis': axis})

    if use_scale:
        scale_factor = st.sidebar.slider("Scale Factor", 0.5, 2.0, 1.2, 0.1)
        methods.append({'method': 'scale', 'scale_factor': scale_factor})

    if use_shift:
        shift_ratio = st.sidebar.slider("Shift Ratio", -0.3, 0.3, 0.1, 0.05)
        methods.append({'method': 'shift', 'shift_ratio': shift_ratio})

    # 증강 실행 버튼
    if st.sidebar.button("Apply Augmentation", type="primary"):
        if len(methods) > 0:
            with st.spinner("Applying augmentation..."):
                st.session_state.augmented_data = augment_dataframe(total_data, methods, augment_per_method)
                st.sidebar.success(f"Augmentation completed!")

                # 통계 표시
                original_count = len(original_data)
                augmented_count = len(st.session_state.augmented_data)
                increase = augmented_count - original_count
                st.sidebar.metric("Total Rows", f"{augmented_count}", f"+{increase}")
        else:
            st.sidebar.warning("Please select at least one augmentation method!")

    # 리셋 버튼 추가
    if st.session_state.augmented_data is not None:
        if st.sidebar.button("Reset Augmentation"):
            st.session_state.augmented_data = None
            st.sidebar.info("Augmentation reset. Showing original data.")
            st.rerun()

    # 저장 버튼
    if st.session_state.augmented_data is not None:
        if st.sidebar.button("Save Augmented Data"):
            # 데이터 소스에 따라 저장 경로 결정
            if data_source == "data1 only (clean)":
                output_folder = r"D:\JJS\Univ_classses\3_2\기계학습\team_project\ml-master\augmentation_data1"
            elif data_source == "data2 only (noisy)":
                output_folder = r"D:\JJS\Univ_classses\3_2\기계학습\team_project\ml-master\augmentation_data2"
            else:  # Original (data1 + data2)
                output_folder = r"D:\JJS\Univ_classses\3_2\기계학습\team_project\ml-master\augmentation_data_combined"

            save_augmented_data(st.session_state.augmented_data, output_folder)
            st.sidebar.success(f"Saved to: {output_folder}")


display_data = st.session_state.augmented_data if st.session_state.augmented_data is not None else total_data

# 메인 화면 
st.title("3D Motion Data Visualization & Augmentation")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualization", "Statistics"])

#Data Overview
with tab1:
    st.header("Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(display_data))
    with col2:
        st.metric("Unique Data IDs", display_data['data_id'].nunique())
    with col3:
        st.metric("Categories", len(CATEGORY_LIST))

    st.subheader("Data Description")
    st.write(
        
    )

    st.subheader("Full Dataset")
    st.dataframe(display_data, use_container_width=True, height=400)

# Visualization
with tab2:
    st.header("3D Motion Visualization")

    # 카테고리 선택
    category = st.selectbox("Select Category", options=CATEGORY_LIST, key="viz_category")

    if category:
        category_df = display_data[display_data["category"] == category]
        data_id_list = sorted(category_df["data_id"].unique())

        # Data ID 선택
        data_id = st.selectbox("Select Data ID", options=data_id_list, key="viz_data_id")

        if data_id:
            # 원본 데이터와 비교 옵션
            show_comparison = st.checkbox("Compare with Original", value=False)

            if show_comparison and st.session_state.augmented_data is not None and data_id >= 1000:
                # 증강된 데이터인 경우 원본과 비교
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Augmented Data")
                    fig = write_data(display_data, category, data_id)
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.subheader("Original Data")
                    original_id = int(str(data_id)[0])  # 1001 -> 1
                    fig_orig = write_data(original_data, category, original_id)
                    st.pyplot(fig_orig)
                    plt.close()
            else:
                # 단일 데이터 시각화
                fig = write_data(display_data, category, data_id)
                st.pyplot(fig)
                plt.close()

            # 데이터 통계 표시
            selected_data = category_df[category_df["data_id"] == data_id]
            st.subheader("Data Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Time Steps", len(selected_data))
            with col2:
                st.metric("X Range", f"{selected_data['x'].min():.1f} ~ {selected_data['x'].max():.1f}")
            with col3:
                st.metric("Y Range", f"{selected_data['y'].min():.1f} ~ {selected_data['y'].max():.1f}")
            with col4:
                st.metric("Z Range", f"{selected_data['z'].min():.1f} ~ {selected_data['z'].max():.1f}")

#Statistics
with tab3:
    st.header("Dataset Statistics")

    # 카테고리별 통계
    st.subheader("Category-wise Data Count")

    category_stats = []
    for cat in CATEGORY_LIST:
        original_count = len(original_data[original_data['category'] == cat]['data_id'].unique())
        current_count = len(display_data[display_data['category'] == cat]['data_id'].unique())
        category_stats.append({
            'Category': cat,
            'Original': original_count,
            'Current': current_count,
            'Increase': current_count - original_count
        })

    stats_df = pd.DataFrame(category_stats)
    st.dataframe(stats_df, use_container_width=True)

    # 막대 그래프
    st.subheader("Visual Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(CATEGORY_LIST))
    width = 0.35

    ax.bar([i - width/2 for i in x], stats_df['Original'], width, label='Original', alpha=0.8)
    ax.bar([i + width/2 for i in x], stats_df['Current'], width, label='Current', alpha=0.8)

    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Data Files')
    ax.set_title('Data Count Comparison by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LIST, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    st.pyplot(fig)
    plt.close()

    # 전체 통계
    st.subheader("Overall Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Data:**")
        st.write(f"- Total Rows: {len(original_data)}")
        st.write(f"- Unique IDs: {original_data['data_id'].nunique()}")

    with col2:
        st.write("**Current Data:**")
        st.write(f"- Total Rows: {len(display_data)}")
        st.write(f"- Unique IDs: {display_data['data_id'].nunique()}")
        if st.session_state.augmented_data is not None:
            increase = len(display_data) - len(original_data)
            st.write(f"- **Increase: +{increase} rows ({increase/len(original_data)*100:.1f}%)**")
