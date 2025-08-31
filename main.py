
import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title=" യ˚*您最齊全的中文客服檢索回覆*˚യ ", page_icon="💡", layout="wide")
st.title("യ˚*您最齊全的中文客服檢索回覆*˚യ")


DEFAULT_FAQ = pd.DataFrame(
    [
        {"question":"你們的營業時間是？","answer":"我們的客服時間為週一至週五 09:00–18:00（國定假日除外）。"},
        {"question":"如何申請退貨？","answer":"請於到貨 7 天內透過訂單頁面點選『申請退貨』，系統將引導您完成流程。"},
        {"question":"運費如何計算？","answer":"單筆訂單滿 NT$ 1000 免運，未滿則酌收 NT$ 80。"},
        {"question":"可以開立發票嗎？","answer":"我們提供電子發票，請於結帳時填寫統一編號與抬頭。"},
        {"question":"訂單大概多久可以出貨？","answer":"一般商品會於付款完成後 2 個工作天內出貨，預購或客製化商品依商品頁說明為主。"},
        {"question":"如何查詢訂單狀態？","answer":"您可以登入會員中心，於『我的訂單』頁面查看最新出貨與配送進度。"},
        {"question":"付款方式有哪些？","answer":"我們提供信用卡、ATM 轉帳、超商代碼與行動支付等多種付款方式。"},
        {"question":"商品有保固嗎？","answer":"大部分商品享有原廠保固，詳細資訊請參考商品頁或保固卡。"},
        {"question":"是否提供海外配送？","answer":"目前僅提供台灣本島與離島配送，海外配送服務尚未開放。"},
        {"question":"客服聯絡方式是什麼？","answer":"您可以透過客服信箱 service@example.com 或撥打 02-1234-5678 與我們聯繫。"},
        {"question":"下單後多久可以發貨？","answer":"大部分商品會於 24 小時內處理出貨，如遇特殊活動商品可能會稍有延遲，請見諒。"},
        {"question":"商品缺貨怎麼辦？","answer":"若商品缺貨，我們會立即主動通知您，您可選擇等待補貨或取消該項商品。"},  # 缺貨處理語術混合整合 :contentReference[oaicite:0]{index=0}
        {"question":"商品實物與網站照片不一樣怎麼辦？","answer":"商品照片因拍攝光線或解析度可能與實物略有差異，若收到商品與預期差異過大，可提供照片，我們將協助處理。」"},  # 網路常見問題 :contentReference[oaicite:1]{index=1}
        {"question":"可以使用折扣碼或優惠券嗎？","answer":"當您在購物車或結帳頁輸入優惠碼後，系統會自動計算折扣。若無法使用，請確認活動期間或聯絡客服。"},
        {"question":"如何追蹤配送進度？","answer":"配送進度可在『我的訂單』頁面查詢，也會透過簡訊／Email 提供物流狀態更新。"},
        {"question":"如何取消訂單？","answer":"若訂單尚未出貨，可於訂單頁面選擇『取消訂單』。若已進入出貨流程，請聯絡客服協助處理。"},
        {"question":"退貨需要運費嗎？","answer":"若是因商品本身瑕疵，我們將負擔退貨運費；若是消費者因素，運費需由您負擔。"},
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
c = st.slider("信心門檻", 0.0, 1.0, 0.2, key="c")

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










