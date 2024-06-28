import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pymysql

# BERT 모델 로드
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 데이터베이스 연결 설정
db_config = {
    'host': "13.124.118.56",
    'user': "test",
    'password': "1111",
    'database': "percol"
}

def fetch_perfume_contents_from_db():
    db = pymysql.connect(**db_config)
    cursor = db.cursor()
    sql = "SELECT id, content FROM perfume"
    cursor.execute(sql)
    db_contents = cursor.fetchall()
    db.close()
    return db_contents

def encode_corpus(contents):
    return model.encode(contents, convert_to_tensor=True)

def find_similar_contents(user_query, corpus_embeddings, contents, top_k=5):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    similar_contents = [(contents[idx], cos_scores[idx].item()) for idx in top_results]
    return similar_contents

def generate_prompt_response(answers):
    """
    주어진 대답들을 합쳐서 프롬프트 질문을 생성합니다.
    
    Parameters:
        answers (list): 1번부터 5번까지의 질문들에 대한 대답 리스트.
        
    Returns:
        str: 합쳐진 프롬프트 질문.
    """
    if len(answers) != 5:
        raise ValueError("5개의 대답이 필요합니다.")
    
    prompt = (
        f"나는 {answers[3]}을 좋아하고 {answers[0]}한 향을 좋아해. 그리고 {answers[1]}을(를) 할 때 뿌릴거야. "
        f"향수의 지속시간은 {answers[4]}이었으면 좋겠고 가격대는 5-10만원대였으면 좋겠어."
    )
    
    return prompt

def main():
    # 프론트엔드에서 받은 대답들
    answers = [
        "플로럴 (꽃향)",  # 1번 질문에 대한 대답
        "직장/학교",      # 2번 질문에 대한 대답
        "우아하고 세련된 기분",  # 3번 질문에 대한 대답
        "봄",            # 4번 질문에 대한 대답
        "5-6시간 (긴 지속)"  # 5번 질문에 대한 대답
    ]

    # 프롬프트 질문 생성
    user_query = generate_prompt_response(answers)
    print(f"Generated Prompt: {user_query}")

    # 데이터베이스에서 콘텐츠 가져오기
    db_contents = fetch_perfume_contents_from_db()
    contents = [content[1] for content in db_contents if content[1] is not None]

    # 코퍼스 임베딩 계산
    corpus_embeddings = encode_corpus(contents)

    # 유사한 콘텐츠 찾기
    similar_contents = find_similar_contents(user_query, corpus_embeddings, contents)

    print("\n입력한 질문:", user_query)
    print("\n유사한 콘텐츠 5가지:")
    for idx, (content, score) in enumerate(similar_contents, 1):
        print(f"{idx}. {content} (Score: {score:.4f})")

if __name__ == "__main__":
    main()
