
import requests
from urllib.parse import urlparse, parse_qs
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import numpy as np
import re #biểu thức chính quy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from underthesea import chunk
from underthesea import word_tokenize

import pickle
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_and_evaluate_svm(data):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'],
        data['label'],
        test_size=0.2,
        random_state=42
    )

    # Vectorize the text data using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Define the SVM model
    svm_model = SVC()

    # Define the parameter grid for GridSearchCV
    param_grid = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

    # Perform GridSearchCV
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
    grid_search_fit = grid_search.fit(X_train_tfidf, y_train)
    
    best_model = grid_search_fit.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)
    # Optionally, you can return the trained model
    return best_model, X_test_tfidf, y_test, vectorizer, y_train, y_pred, grid_search_fit


def word(text):
    word_tokenize(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    chunks = chunk(text)
    filtered_chunks = [chunk[0] for chunk in chunks if chunk[2] != 'O']
    result_sentence = ' '.join(filtered_chunks)
    return result_sentence


def evaluate_model_performance(best_model, X_test_tfidf, y_test, y_pred):
    print("ĐÁNH GIÁ ĐỘ CHI TIẾT CỦA MÔ HÌNH")
    print("-------------------------------------------------------------------------------------")

    # Evaluate accuracy and ROC-AUC score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Độ Chính Xác: {accuracy}')
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC-AUC Score: {roc_auc}")

    print("-------------------------------------------------------------------------------------")
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test_tfidf)

    # Calculate and print classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    print("-------------------------------------------------------------------------------------")

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("-------------------------------------------------------------------------------------")

# Lấy những bài báo random gần đây nhất (Vietnam News API)
api_urls = [
    
    "https://newsdata.io/api/1/news?country=vi&category=top&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=sports&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=technology&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=science&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=entertainment&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=health&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=world&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=politics&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    "https://newsdata.io/api/1/news?country=vi&category=environment&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a"
    "https://newsdata.io/api/1/news?country=vi&category=food&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a"
]
# Danh sách các địa chỉ API
    # TopHot: "https://newsdata.io/api/1/news?country=vi&category=top&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Thể thao: "https://newsdata.io/api/1/news?country=vi&category=sports&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Công Nghệ: "https://newsdata.io/api/1/news?country=vi&category=technology&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Khoa học: "https://newsdata.io/api/1/news?country=vi&category=science&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Giải trí: "https://newsdata.io/api/1/news?country=vi&category=entertainment&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Sức khỏe: "https://newsdata.io/api/1/news?country=vi&category=health&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Thế giới: "https://newsdata.io/api/1/news?country=vi&category=world&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Chính trị: "https://newsdata.io/api/1/news?country=vi&category=politics&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a",
    # Môi trường: "https://newsdata.io/api/1/news?country=vi&category=environment&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a"
    # Food: "https://newsdata.io/api/1/news?country=vi&category=food&apikey=pub_349954ee9077f2ddc34d6b922c7de34859a0a"


user_choice = input("Bạn có muốn thực hiện lấy dữ liệu từ API và cập nhật file CSV không? (y/n): ")

if user_choice.lower() == 'y':
    # Khởi tạo DataFrame trống
    df = pd.DataFrame(columns=["article_id", "title", "link", "creator", "description", "content", "source_id", "category"])

    progress_bar = tqdm(total=len(api_urls), desc="Đang lấy dữ liệu từ API", leave=True)

    for api_url in api_urls:
        try:
            # Gửi yêu cầu GET đến API và nhận dữ liệu JSON
            response = requests.get(api_url)
            data = response.json()

            # Lấy danh sách bài báo từ trường "results"
            results = data.get("results", []) 

            # Nối kết quả từ mỗi API vào DataFrame
            df = pd.concat([df, pd.DataFrame(results, columns=df.columns)], ignore_index=True)

        except Exception as e:
            print(f"Lấy dữ liệu không thành công. Hãy kiểm tra lại {api_url}. Error: {str(e)}")

        # Cập nhật thanh tiến trình
        progress_bar.update(1)

    # Đóng thanh tiến trình
    progress_bar.close()

    df.to_csv('data/RealNewsfrom_NewsData_io_UPDATED.csv', index=False, encoding='utf-8', mode='a', header=False)

    df = pd.read_csv('data/RealNewsfrom_NewsData_io_UPDATED.csv')
    duplicate_rows = df[df.duplicated('article_id')]
    df = df.drop_duplicates('article_id', keep='first')
    df.to_csv('data/RealNewsfrom_NewsData_io_UPDATED.csv', index=False)
    print("Dữ liệu đã được cập nhật thành công.")
else:
    print("Bạn đã chọn không thực hiện cập nhật dữ liệu.")



print("-------------------------------------------------------------------------------------")
#Máy học ------------------------------------------------------------------------------------------------------------------

real_data = pd.read_csv('data/RealNewsfrom_NewsData_io_UPDATED.csv', encoding='utf-8')
fake_data = pd.read_csv('data/FakeNews.csv', encoding='utf-8')

real_data['label'] = 0 #real
fake_data['label'] = 1 #fake

# Kết hợp hai DataFrame thành một DataFrame lớn
data = pd.concat([real_data, fake_data], ignore_index=True)
data = data.fillna('')

# Gộp các trường title, content, creator lại thành 1 trường text (1 cột trong dataframe)
data['text'] = data['title']+ ' ' + data['content'] + ' ' + data['creator']


# Mở tệp dữ liệu temp_data_noClean.csv 
data_check = pd.read_csv('data/temp_data_noClean.csv', encoding='utf-8')


# Tối ưu dữ liệu nhập vào tránh phải chuẩn hóa các bài báo cũ:
# - Bước 1: Ghép (merge) hai DataFrame data và data_check dựa trên cột 'text' với kiểu ghép là 'outer' (ghép toàn bộ).
# - Bước 2: Lọc những hàng chỉ xuất hiện trong data mà không xuất hiện trong data_check.
# - Bước 3: Kiểm tra missing_rows_data_in_datacheck có giá trị được truyền vào hay không:
#     + Nếu không phát hiện không có giá trị nào được truyền vào thì trả về 1 dataframe rỗng chứa duy nhất các trường mặc định của data
#     + Nếu có giá trị được truyền vào missing_rows_data_in_datacheck thức hiện các mục:

merged_data_datacheck = pd.merge(data, data_check, on='text', how='outer', indicator=True) # 

# Lọc các hàng chỉ xuất hiện trong data
missing_rows_data_in_datacheck = merged_data_datacheck[merged_data_datacheck['_merge'] == 'left_only']

# Kiểm tra xem missing_rows_data_in_datacheck có tồn tại hay không
if missing_rows_data_in_datacheck.empty:
    result_value = result_value = pd.DataFrame(columns=['text', 'label'])
else:
    # Nếu có, thực hiện các bước tiếp theo
    data_check = data.copy()                                        #-> copy toàn bộ dữ liệu từ dataframe của data để update data_check (thêm những bài báo mới)
    data_check.to_csv('data/temp_data_noClean.csv', index=False)    #-> đẩy data_check thành 1 file csv để thay thế file cũ (chứa những bài báo cũ lần đã cập nhật)
    temp = missing_rows_data_in_datacheck.copy()                    #-> copy những bài báo data có mà data_check không có vào biến temp
    temp.rename(columns={'label_x': 'label'}, inplace=True)         #-> thay đổi tên trường trong dataframe temp cho phù hợp với dữ liệu đúng
    selected_columns = ['text', 'label']                            #-> tạo một biến mang 2 trường là text và label
    temp_selected = temp[selected_columns]                          #-> tạo 1 dataframe temp_selected chỉ mang 2 trường là text và label của temp
    result_value = temp_selected                                    #-> trả về giá trị cuối là result_value

# In kết quả hoặc sử dụng result_value theo nhu cầu
best_model, X_test_tfidf, y_test, vectorized, y_train, y_pred, grid_search_fit = tqdm(train_and_evaluate_svm(data))

print('Những bài báo mới được cập nhật:')
if result_value.empty:
    print(f"{Fore.RED}Không có nội dung cần cập nhật.{Style.RESET_ALL}")
    evaluate_model_performance(best_model, X_test_tfidf, y_test, y_pred)
else:
    print(result_value)
    print(f"{Fore.GREEN}Đang chuẩn hóa dữ liệu các bài báo mới...{Style.RESET_ALL}")

    tqdm.pandas()
    # result_value.loc[:, 'text'] = result_value['text'].swifter.apply(word).copy()
    result_value.loc[:, 'text'] = result_value['text'].progress_apply(word)
    
    data_have_clean = pd.read_csv('data/temp_data_haveClean.csv', encoding='utf-8')
    data = pd.concat([data_have_clean, result_value[['text', 'label']]], ignore_index=True)
    data = data.fillna('')
    data.to_csv('data/temp_data_haveClean.csv', index=False, encoding='utf-8', mode='a', header=False)
    data[['text', 'label']]
    
    # In ra các tham số tốt nhất
    print('-------------------------------------------------------------------------------------')
    print('Thông số tốt nhất cho mô hình:', grid_search_fit.best_params_)

    with open('temp/vectorized.pkl', 'wb') as f:
        pickle.dump(vectorized, f)  
    with open('temp/svm_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('temp/word_clean.pkl', 'wb') as f:
        pickle.dump(word, f)
    #đánh giá mô hình hiện tại
    
    evaluate_model_performance(best_model, X_test_tfidf, y_test, y_pred)


def main():
    # Yêu cầu người dùng ấn Enter để thoát
    while True:
        user_input = input("Nhấn Enter để thoát.")
        if user_input == "":
            break
        else:
            print("Vui lòng chỉ ấn Enter.")

if __name__ == "__main__":
    main()




