
import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="中文客服檢索回覆", page_icon="💬", layout="wide")
st.title("中文客服檢索回覆")


DEFAULT_FAQ = pd.DataFrame(
    [
        {"question":"你們的營業時間是？","answer":"我們的客服時間為週一至週五 09:00–18:00（國定假日除外）。"},
        {"question":"如何申請退貨？","answer":"請於到貨 7 天內透過訂單頁面點選『申請退貨』，系統將引導您完成流程。"},
        {"question":"運費如何計算？","answer":"單筆訂單滿 NT$ 1000 免運，未滿則酌收 NT$ 80。"},
        {"question":"可以開立發票嗎？","answer":"我們提供電子發票，請於結帳時填寫統一編號與抬頭。"},
    ]
)

if "faq_df" not in st.session_state:
    st.session_state.faq_df = DEFAULT_FAQ.copy()
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None

# 步驟一
st.subheader("上傳知識庫")
uploader = st.file_uploader("限上傳 CSV 檔案", type=["csv"])
if uploader is not None: #若有上傳檔案，內容不是空的 #注意大寫None
    df = pd.read_csv(uploader)
    #st.write(df)
    #取代前面的DEFAULT_FAQ，去空白紀錄
    st.session_state.faq_df = df.dropna().reset_index(drop=True) 
    #df.dropna()刪除空白 reset_index(drop=True)資料重整理
    st.success(f"已成功載入 {len(df)} 筆資料！")

with st.expander("檢視資料", expanded=False): #False是欄位/資料收起來
    st.dataframe(st.session_state.faq_df,use_container_width=True)
    
# 步驟二：建立索引
do_index = st.button("建立/重設索引")

def jieba_tokenize(text:str):
    """使用 jieba 進行中文分詞"""
    return list(jieba.cut(text)) #將傳進來的句子分詞

if do_index or (st.session_state.vectorizer is None): #如果按鈕按下(新增)或原本頁面沒紀錄(剛開)
    corpus = (st.session_state.faq_df["question"].astype(str) + 
              " " +
              st.session_state.faq_df["answer"].astype(str)).tolist()
    v = TfidfVectorizer(tokenizer=jieba_tokenize) #Tfidf統計頻率去當索引
    tfidf = v.fit_transform(corpus)
    st.session_state.vectorizer = v
    st.session_state.tfidf = tfidf 
    st.success("索引建立成功！")

# 步驟三：詢問客服
q = st.text_input("請輸入您的問題：", placeholder="例如：你們的營業時間是？")
top_k = st.slider("選擇回覆數量", 1, 5, 3) #取得前k筆回答
c = st.slider("信心門檻", 0.0, 1.0, 0.5, key="c")

if st.button("送出") and q.strip(): #q.strip()去掉前後的空白 #有按下按鈕且有輸入問題(不是空白)
#python中空白都是false
    if (st.session_state.vectorizer is None) or (st.session_state.tfidf is None):
        st.warning("尚未建立索引，會自動建立。請稍候...")
        corpus = (st.session_state.faq_df["question"].astype(str) + 
              " " +
              st.session_state.faq_df["answer"].astype(str)).tolist()
        v = TfidfVectorizer(tokenizer=jieba_tokenize) #Tfidf統計頻率去當索引
        tfidf = v.fit_transform(corpus)
        st.session_state.vectorizer = v
        st.session_state.tfidf = tfidf 
        st.success("索引建立成功！")

    vec = st.session_state.vectorizer.transform([q]) #針對問題轉換
    sims = linear_kernel(vec, st.session_state.tfidf).flatten() #計算題目跟問題的相似度 看哪個答案符合機率大
    idxc = sims.argsort()[::-1][:top_k] #由大到小排序，取出前top_k筆數量
    rows = st.session_state.faq_df.iloc[idxc].copy() #取出對應的行
    rows['score'] = sims[idxc] #將相似度分數加入


    best_ans = None #預設最佳答案為None (空跟0是不同的)
    #取最高的概率
    best_score = float(rows['score'].iloc[0]) if len(rows) > 0 else 0.0 # >0不一定要寫 一樣意思
    #長度若大於0，取第一筆的分數，否則為0.0
    if best_score >= c: #如果最高分數大於信心門檻
        best_ans = rows['answer'].iloc[0]
        # 0就是false 不是0就是true

    if best_ans:
        st.success(f"最佳回覆：{best_ans}")
    else:
        st.warning("未找到符合的回覆。")
        st.info("找不到符合的回覆，請嘗試更換問題或調整信心門檻。")

    #展開可能的回答
    with st.expander("檢索結果：", expanded=False): #一開始不要展開
         st.dataframe(rows[['question', 'answer', 'score']], use_container_width=True) #會根據畫面縮放



