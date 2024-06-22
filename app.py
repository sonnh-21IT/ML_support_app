import pandas as pd # type: ignore
import numpy as np # type: ignore
import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import os

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression,LogisticRegression # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.metrics import r2_score,accuracy_score # type: ignore
from sklearn import tree # type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn import metrics # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore

# from surprise import Dataset, Reader # type: ignore
# from surprise import SVD # type: ignore
# from surprise import accuracy # type: ignore

# Function to load CSV file with specified encoding
def load_csv(file, encoding='utf-8'):
    df = pd.read_csv(file, encoding=encoding)
    return df

def plot_linear_regression(X_test, y_test, feature_columns, target_column):
    # Tạo DataFrame từ X_test và y_test
    df = pd.DataFrame(X_test, columns=feature_columns)
    df[target_column] = y_test

    # Vẽ đồ thị đa biến
    sns.pairplot(df, x_vars=feature_columns, y_vars=target_column, kind='reg')

    # Hiển thị biểu đồ trong Streamlit
    fig = plt.gcf()  # Lấy đối tượng hình ảnh (figure) hiện tại
    st.pyplot(fig)

def plot_logistic_regression(X_test, y_test, model, feature_columns, target_column):
    y_pred = model.predict(X_test)

    # Tạo DataFrame từ X_test và y_test
    df = pd.DataFrame(X_test, columns=feature_columns)
    df[target_column] = y_test

    # Vẽ đồ thị đa biến với đường hồi quy
    sns.pairplot(df, x_vars=feature_columns, y_vars=target_column, kind='reg')

    # Hiển thị biểu đồ trong Streamlit
    fig = plt.gcf()  # Lấy đối tượng hình ảnh (figure) hiện tại
    st.pyplot(fig)

def plot_decision_tree(classifier, feature_names, class_names):
    feature_names = list(map(str, feature_names))
    class_names = list(map(str, class_names))
    fig, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(classifier, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, ax=ax)
    st.pyplot(fig)

def plot_random_forest_class(X_test, y_test, model):
    # Thực hiện dự đoán trên dữ liệu kiểm tra
    y_pred = model.predict(X_test)
    
    # Tạo biểu đồ
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Dự đoán')
    ax.set_ylabel('Thực tế')
    ax.set_title('Ma trận confusion - Random Forest Classifier')
    
    # Hiển thị biểu đồ trong ứng dụng Streamlit
    st.pyplot(fig)

def plot_random_forest_reg(X_test, y_test, model):
    # Thực hiện dự đoán trên dữ liệu kiểm tra
    y_pred = model.predict(X_test)
    
    # Tạo biểu đồ
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Thực tế')
    ax.set_ylabel('Dự đoán')
    ax.set_title('Biểu đồ Random Forest Regression')
    
    # Hiển thị biểu đồ trong ứng dụng Streamlit
    st.pyplot(fig)

def plot_recommendation_system(results):
    items = results.keys()
    scores = results.values()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(items, scores)
    ax.set_xlabel('Sản phẩm')
    ax.set_ylabel('Điểm gợi ý')
    ax.set_title('Kết quả hệ thống gợi ý')
    ax.set_xticklabels(items, rotation=90)
    st.pyplot(fig)

def plot_knn(X_train,X_test,y_train,y_test,model):
    # Khởi tạo danh sách số lượng hàng xóm (neighbors) và độ chính xác (accuracy)
    k_values = []
    accuracy_values = []

    # Lặp qua các số lượng hàng xóm và tính độ chính xác tương ứng
    for k in range(1, 11):
        model.n_neighbors = k
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # Lưu giá trị vào danh sách
        k_values.append(k)
        accuracy_values.append(accuracy)

    # Vẽ biểu đồ
    fig, ax = plt.subplots()
    ax.plot(k_values, accuracy_values, marker='o')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Accuracy')
    ax.set_title('KNN Model Performance')
    ax.grid(True)

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

def plot_feature_scatter(df1, df2, features):
    for feature in features:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df1[feature], df2[feature], marker='+')
        ax.set_xlabel(feature)
        ax.set_ylabel(feature)
        plt.tight_layout()
        st.pyplot(fig)

def plot_target_countplot(target_column):
    fig, ax = plt.subplots()
    sns.countplot(data=st.session_state.df, x=target_column)
    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    ax.set_title("Countplot")
    ax.tick_params(axis='x')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')
    st.pyplot(fig)

# Biểu đồ phân phối
def plot_distribution(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data, ax=ax)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution Plot')
    st.pyplot(fig)

# Biểu đồ tương quan
def plot_correlation(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Plot')
    st.pyplot(fig)

# Biểu đồ hộp
def plot_boxplot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, ax=ax)
    plt.ylabel('Value')
    plt.title('Box Plot')
    st.pyplot(fig)

# Biểu đồ tần suất
def plot_frequency(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data, ax=ax)
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title('Frequency Plot')
    st.pyplot(fig)

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    return model, accuracy

def train_logistic_regression(X_train, y_train, X_test, y_test):
    cols = X_train.columns
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def train_decision_tree_class(X_train, y_train, X_test, y_test, feature_columns, draw_tree=True):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def train_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def train_random_forest_class(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy
def train_random_forest_reg(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error_score = metrics.r2_score(y_test, y_pred)

    return model, error_score

# def train_recommendation_system(data):
#     # Đọc dữ liệu vào định dạng Surprise Dataset
#     reader = Reader(rating_scale=(1, 5))
#     dataset = Dataset.load_from_df(data, reader)

#     # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
#     train_set, test_set = train_test_split(dataset, test_size=0.2)

#     # Xây dựng mô hình SVD
#     model = SVD()

#     # Huấn luyện mô hình trên tập huấn luyện
#     model.fit(train_set)

#     # Đánh giá mô hình trên tập kiểm tra
#     predictions = model.test(test_set)
#     accuracy.rmse(predictions)

#     return model

def train_model(select_model, feature_columns, target_column):
    X_train, X_test, y_train, y_test = train_test_split(st.session_state.df[feature_columns], st.session_state.df[target_column], test_size=0.2, random_state=42)
    if select_model == "Linear Regression":
        model, accuracy = train_linear_regression(X_train, y_train, X_test, y_test)
        st.write(f"Độ Chính Xác Của Mô Hình Là: {accuracy}")
        plot_linear_regression(X_test,y_test,feature_columns, target_column)
    elif select_model == "Logistic Regression":
        model, accuracy = train_logistic_regression(X_train, y_train, X_test, y_test)
        st.write(f"Độ Chính Xác Của Mô Hình Là: {accuracy}")
        plot_logistic_regression(X_test,y_test,model,feature_columns, target_column)
    elif select_model == "Decision Tree Classifier":
        model, accuracy = train_decision_tree_class(X_train, y_train, X_test, y_test, feature_columns)
        st.write(f"Độ Chính Xác Của Mô Hình Là: {accuracy}")
        plot_decision_tree(model, feature_columns, model.classes_)
    elif select_model == "K-nearest Neighbors":
        model, accuracy = train_knn(X_train, y_train, X_test, y_test)
        st.write(f"Độ Chính Xác Của Mô Hình Là: {accuracy}")
        plot_knn(X_train,X_test,y_train,y_test,model)
    elif select_model == "Random Forest Classifier":
        model, accuracy = train_random_forest_class(X_train, y_train, X_test, y_test)
        st.write(f"Độ Chính Xác Của Mô Hình Là: {accuracy}")
        plot_random_forest_class(X_test,y_test,model)
    elif select_model == "Random Forest Regression":
        model, accuracy = train_random_forest_reg(X_train, y_train, X_test, y_test)
        st.write(f"Độ Chính Xác Của Mô Hình Là: {accuracy}")
        plot_random_forest_reg(X_test,y_test,model)
    else:
        st.write("Lựa chọn mô hình không hợp lệ.")
        return
    
    return model

def func_main():
    st.title("Trình tải lên CSV và Bộ chọn cột")
    uploaded_file = st.file_uploader("Chọn một tệp CSV", type=['csv'])

    if uploaded_file is not None:
        df = load_csv(uploaded_file, encoding='latin1')
        if 'df' not in st.session_state:
            st.session_state['df'] = df
        # Display DataFrame in fullscreen mode
        if 'df' in st.session_state:
            with st.expander("Xem tất cả dữ liệu", expanded=True):            
                st.dataframe(st.session_state.df)

                rows, cols = st.session_state.df.shape

                st.write("Hàng: " + str(rows))
                st.write("Cột: " + str(cols))
                st.write("Giá trị null: " + str(st.session_state.df.isnull().any(axis=1).sum()))
                if st.session_state.df.isnull().any(axis=1).sum() > 0:
                    if st.button("Xóa các hàng có giá trị null"):
                        st.session_state.df.dropna(inplace = True)
                        st.rerun()
                st.write("Hàng trùng lặp: " + str(st.session_state.df.duplicated().sum()))
                if st.session_state.df.duplicated().sum() > 0:
                    if st.button("Xóa các hàng trùng lặp"):
                        st.session_state.df.drop_duplicates(inplace = True)
                        st.rerun()
                st.write("Thông tin dữ liệu:")
                st.write(st.session_state.df.describe())
                st.write("Kiểu dữ liệu:")
                st.write(st.session_state.df.dtypes.to_frame().T)
                st.write("Số hàng không có dữ liệu:")
                st.write(st.session_state.df.isnull().sum(axis=0).to_frame().T)
                st.write("Biến phân loại:")

                categorical_columns_dtype = [col for col in st.session_state.df.select_dtypes(include='object').columns if st.session_state.df[col].apply(type).eq(str).all()]
                categorical_columns_unique = [col for col in st.session_state.df.select_dtypes(include='object').columns if st.session_state.df[col].nunique() < 10]
                final_categorical_columns = list(set(categorical_columns_dtype) | set(categorical_columns_unique))
                st.write(final_categorical_columns)

                select_column = st.selectbox("Chọn cột:",st.session_state.df.columns)
                if st.button("Hiển thị giá trị duy nhất"):
                    st.write(st.session_state.df[select_column].value_counts().to_frame().T)

                dfContainer = st.container()
                dfLeft, dfRight = dfContainer.columns(2)
                with dfLeft:
                    optionL = st.selectbox(
                        "Chọn cột",
                        st.session_state.df.columns
                    )
                with dfRight:
                    optionR = st.selectbox(
                        "Chọn thao tác",
                        ("Delete","To numeric","To object","To Datetime","Missing to mean","Missing to median","Missing to mode","Missing to value","Value to value","Rename","Handle monetary value")
                    )
                
                is_two_col = False
                if optionR == "Rename":
                    new_col_name = st.text_input("New column name")
                elif optionR == "Null to value":
                    new_value_for_null = st.text_input("New value")
                elif optionR == "Value to value":
                    with dfLeft:
                        old_value_for_value = st.text_input("Old value")
                    with dfRight:
                        new_value_for_value = st.text_input("New value")
                elif optionR == "Handle monetary value":
                    if st.session_state.df[optionL].dtype == object:
                        for index, row in st.session_state.df.iterrows():
                            value = row[optionL]
                            if isinstance(value, str) and '-' in value:
                                is_two_col = True
                                break
                        if is_two_col:
                            with dfLeft:
                                new_col_name_min = st.text_input("Min column")
                            with dfRight:
                                new_col_name_max = st.text_input("Max column")
                else:
                    st.write("Thao tác này chưa khả dụng. Vui lòng thử lại sau!")
                if "info" in st.session_state:
                    st.write(st.session_state.info)
                    st.session_state.info = ""
                if st.button("Thực thi"):
                    if optionR == "To numeric":
                        if st.session_state.df[optionL].dtype == object:
                            st.session_state.df[optionL] = pd.factorize(st.session_state.df[optionL])[0]
                        else:
                            st.session_state.info = "Cột hiện tại đang có kiểu dữ liệu là số."
                    elif optionR == "To object":
                        if st.session_state.df[optionL].dtype != object:
                            st.session_state.df[optionL] = st.session_state.df[optionL].astype(object)
                        else:
                            st.session_state.info = "Kiểu dữ liệu hiện tại và kiểu dữ liệu mới trùng nhau."
                    elif optionR == "To Datetime":
                        st.session_state.df[optionL] = pd.to_datetime(st.session_state.df[optionL])
                    elif optionR == "Rename":
                        st.session_state.df.rename(columns={optionL : new_col_name}, inplace=True)
                    elif optionR == "Missing to value":
                        st.session_state.df[optionL].fillna(new_value_for_null, inplace=True)
                    elif optionR == "Value to value":
                        st.session_state.df[optionL].replace(old_value_for_value, new_value_for_value, inplace=True)
                    elif optionR == "Missing to mean":
                        if st.session_state.df[optionL].dtype == 'object':
                            st.session_state.info = "Với kiểu dữ liệu là object thì chỉ có thể thay thế bằng giá trị mode"
                        else:
                            mean = df[optionL].mean()
                            st.session_state.df[optionL].fillna(mean, inplace=True)
                    elif optionR == "Missing to median":
                        if st.session_state.df[optionL].dtype == 'object':
                            st.session_state.info = "Với kiểu dữ liệu là object thì chỉ có thể thay thế bằng giá trị mode"
                        else:
                            median = st.session_state.df[optionL].median()
                            st.session_state.df[optionL].fillna(median, inplace=True)
                    elif optionR == "Missing to mode":
                        mode = st.session_state.df[optionL].mode()
                        st.session_state.df[optionL].fillna(mode, inplace=True)
                    elif optionR == "Handle monetary value":
                        if is_two_col:
                            st.session_state.df[new_col_name_min] = np.nan
                            st.session_state.df[new_col_name_max] = np.nan
                            for index, row in st.session_state.df.iterrows():
                                if '-' in str(row[optionL]):
                                    min_price, max_price = str(row[optionL]).split('-')
                                    st.session_state.df.at[index, new_col_name_min] = float(min_price.replace('$', '').replace(',', ''))
                                    st.session_state.df.at[index, new_col_name_max] = float(max_price.replace('$', '').replace(',', ''))
                        else:
                            for index, row in st.session_state.df.iterrows():
                                st.session_state.df.at[index, optionL] = float(str(row[optionL]).replace('$', '').replace(',', ''))
                    elif optionR == "Delete":
                        st.session_state.df.drop(columns=[optionL], inplace=True)
                   
                    st.rerun()
                
                if st.button("Hiển thị biểu đồ tương quan"):
                    numeric_columns = df.select_dtypes(include='number')
                    correlation_matrix = numeric_columns.corr()
                    # Vẽ heatmap
                    fig, ax = plt.subplots(figsize=(16, 12))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, ax=ax)
                    plt.title('Ma trận tương quan giữa các biến số')

                    # Hiển thị biểu đồ trong Streamlit
                    st.pyplot(fig)
            
            dt_select_container = st.container()
            dt_select_left, dt_select_right = dt_select_container.columns(2)

            target_column = dt_select_left.selectbox("Chọn biến phụ thuộc x:", st.session_state.df.columns)
            feature_columns = dt_select_right.multiselect("Chọn biến độc lập y:", st.session_state.df.columns)

            # Process data
            target_data = st.session_state.df[target_column]  # Select the target column
            feature_data = st.session_state.df[feature_columns]   # Select the feature columns

            # Optionally, you can display processed data
            dt_container = st.container()
            dt_left, dt_right = dt_container.columns(2)
            dt_left.write("Dữ liệu biến phụ thuộc đã xử lý:")
            dt_left.dataframe(target_data)

            dt_right.write("Dữ liệu biến độc lập đã xử lý:")
            dt_right.dataframe(feature_data)

            chart_container = st.container()
            c_left, c_center = chart_container.columns(2)
            chart_type = c_left.selectbox("Chọn loại biểu đồ:",("Biểu đồ phân phối","Biểu đồ tương quan","Biểu đồ hộp","Biểu đồ tần suất"))
            if chart_type in ("Biểu đồ phân phối","Biểu đồ hộp") and len(feature_columns) > 0:
                select_column_feature = c_center.selectbox("Chọn biến độc lập:",feature_columns)
            if st.button("Vẽ biểu đồ"):
                if len(feature_columns) > 0:
                    if chart_type == "Biểu đồ phân phối":
                        plot_distribution(st.session_state.df[select_column_feature])
                    elif chart_type == "Biểu đồ tương quan":
                        plot_correlation(st.session_state.df[feature_columns + [target_column]])
                    elif chart_type == "Biểu đồ hộp":
                        plot_boxplot(st.session_state.df[select_column_feature])
                    else: 
                        st.write("Biểu đồ chưa khả dụng. Vui lòng thử lại sau!")
                else:
                    st.write("Danh sách cột biến độc lập còn trống")

            b_train_container = st.container()
            bt_left, bt_right = b_train_container.columns(2)
            select_value_vis = bt_left.selectbox("Chọn tập dữ liệu để hiển thị:",("Biến phục thuộc x","Biến độc lập y"))

            if bt_left.button("Hiển thị"):
                if select_value_vis == "Biến phục thuộc x":
                    plot_target_countplot(target_column=target_column)
                elif select_value_vis == "Biến độc lập y":
                    train_df, test_df = train_test_split(st.session_state.df, test_size=0.2, random_state=42)
                    min_rows = min(len(train_df), len(test_df))
                    train_df = train_df[:min_rows]
                    test_df = test_df[:min_rows]
                    plot_feature_scatter(train_df[:], test_df[:], feature_data)
            if len(feature_columns) > 0:
                select_model = bt_right.selectbox("Chọn mô hình:", ("Linear Regression","Logistic Regression","Decision Tree Classifier","K-nearest Neighbors","Random Forest Classifier","Random Forest Regression","Recommendation System"))
                if bt_right.button("Chạy mô hình"):
                    st.session_state.model = train_model(select_model, feature_columns, target_column)
            if 'model' in st.session_state:
                st.write("Nhập các giá trị để dự đoán dựa trên mô hình:")
                input_values = []
                exception_inputs = []
                fill_container = st.container()
                fill_left,fill_right = fill_container.columns(2)
                for index, fill in enumerate(feature_columns):
                    # Chọn đối tượng text_input dựa trên chỉ mục
                    text_input_object = fill_left if index == 0 or index % 2 == 0 else fill_right

                    # Sử dụng text_input để nhận giá trị từ người dùng và lưu vào biến 'input_value'
                    input_value = text_input_object.text_input(fill)

                    # Kiểm tra và xử lý giá trị nhập vào
                    try:
                        float_value = float(input_value)
                        input_values.append((fill, float_value))
                    except ValueError:
                        exception_inputs.append(fill)
                if len(exception_inputs) > 0:
                    invalid_values = ", ".join(exception_inputs)
                    error_message = "Giá trị không hợp lệ: {}. Vui lòng nhập số nguyên hoặc số thực.".format(invalid_values)
                    st.write(error_message)
                else: 
                    if st.button("Dự đoán"):
                        input_array = np.array([input_value[1] for input_value in input_values]).reshape(1, -1)
                        predictions = st.session_state.model.predict(input_array)
                        st.write(predictions[0])


def merge_datasets(dataset1, dataset2, common_column, merge_type):
    if merge_type == "Chiều rộng":
        merged_data = pd.merge(dataset1, dataset2, on=common_column)
    elif merge_type == "Chiều sâu":
        if list(dataset1.columns) == list(dataset2.columns):
            merged_data = pd.concat([dataset1, dataset2], axis=0, ignore_index=True)
        else:
            st.warning("Tên cột và thứ tự cột của hai dataset phải giống nhau để thực hiện nối theo chiều sâu.")
            merged_data = None
    return merged_data

def func_merger():
    st.title("Ứng dụng nối (merge) dataset")

    # Tải lên các file CSV
    uploaded_files = st.file_uploader("Chọn các file CSV", accept_multiple_files=True, type="csv")

    datasets = []
    dataset_names = []
    for file in uploaded_files:
        dataset = pd.read_csv(file)
        datasets.append(dataset)
        dataset_names.append(os.path.splitext(file.name)[0])

    common_columns = None
    # Hiển thị thông tin về các dataset đã tải lên
    st.header("Các dataset đã tải lên:")
    for i, dataset in enumerate(datasets):
        st.subheader(f"Dataset {dataset_names[i]}")
        st.write(dataset)
        st.write(dataset.shape)
        if common_columns is None:
            common_columns = set(dataset.columns)  # Gán danh sách cột chung ban đầu
        else:
            common_columns &= set(dataset.columns)  # Cập nhật danh sách cột chung

    # Kiểm tra nếu có ít nhất 2 dataset để nối
    if len(datasets) >= 2:
        # Chọn kiểu nối (theo chiều rộng hoặc chiều sâu)
        merge_type = st.radio("Chọn kiểu nối", options=["Chiều rộng", "Chiều sâu"])
        dataset1_index = 0
        dataset2_index = 0
        common_column = ""
        if merge_type == "Chiều rộng":
            # Chọn dataset và cột chung để nối
            dataset1_name = st.selectbox("Chọn dataset 1", dataset_names)
            dataset2_name = st.selectbox("Chọn dataset 2", dataset_names)
            
            dataset1_index = dataset_names.index(dataset1_name)
            dataset2_index = dataset_names.index(dataset2_name)
            common_column = st.selectbox("Chọn cột chung để nối", options=common_columns)

        # Nút để thực hiện nối (merge) các dataset
        if st.button("Nối (Merge) Dataset"):
            merged_data = merge_datasets(datasets[dataset1_index], datasets[dataset2_index], common_column, merge_type)
            st.header("Dataset sau khi nối:")
            st.write(merged_data)
            st.write(merged_data.shape)
    else:
        st.warning("Bạn cần tải lên ít nhất 2 dataset để thực hiện nối (merge).")

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox("Điều hướng", ["Trang chủ","Nối dữ liệu"])

    if page == "Trang chủ":
        func_main()
    elif page == "Nối dữ liệu":
        func_merger()

if __name__ == "__main__":
    main()
