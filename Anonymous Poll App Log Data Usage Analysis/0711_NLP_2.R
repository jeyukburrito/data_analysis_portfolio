# R 토픽 모델링 전체 워크플로우
# polls_question.csv 기반

# 1. 패키지 로드 및 설치
required_pkgs <- c(
  "readr", "dplyr", "udpipe", "tidytext", "topicmodels",
  "textmineR", "Matrix", "ggplot2"
)
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

setwd("C:/Users/Yoo/Desktop/codeit/ProjectData/Data_고급/polls")

# 2. 데이터 로딩
df    <- read_csv("polls_question.csv")

# "vote" 텍스트를 포함한 문서 제외
if ("vote" %in% df$question_text) {
  df <- df %>% filter(question_text != "vote")
}

texts <- df$question_text


# 3. 형태소 분석 (udpipe 한국어 모델)
ud_model <- udpipe_download_model(language = "korean")
ud      <- udpipe_load_model(ud_model$file_model)
anno    <- udpipe_annotate(ud, x = texts, doc_id = seq_along(texts))
anno_df <- as.data.frame(anno)

# 4. 전처리: 명사 추출, 어근(lemma)만, + 이후 조사 제거, 소문자, 길이 필터
library(stringr)
# 4. 전처리: 명사 추출, 어근 추출, + 이후 조사 제거, 소문자, 한글·영어·숫자 외 문자 제거, 길이 필터
noun_df <- anno_df %>%
  filter(upos == "NOUN") %>%
  mutate(
    word = str_extract(lemma, "^[^\\+]+"),       # + 뒤 조사 제거
    word = tolower(word),                           # 소문자 변환
    word = str_replace_all(word, "[^가-힣a-zA-Z0-9]", "")  # 한글·영어·숫자 외 문자 제거
  ) %>%
  filter(str_detect(word, "^[가-힣a-zA-Z0-9]{2,}$"))  # 한글, 영어, 숫자 조합 2자 이상 추출

# 5. 불용어(stopwords) 및 고빈도 단어 제거
custom_stop <- c("수","것","등","들","안","더","는",
                                "사람","때","날","수","나","말","매일", "를", "을", "친구")

noun_df <- noun_df %>%
  filter(!word %in% custom_stop)

doc_count <- length(unique(noun_df$doc_id))
high_df   <- noun_df %>%
  distinct(doc_id, word) %>%
  count(word, name = "df") %>%
  filter(df / doc_count > 0.3) %>%
  pull(word)
noun_df  <- noun_df %>% filter(!word %in% high_df)

# 6. DTM 생성 (단어 출현 카운트)
count_df <- noun_df %>% count(doc_id, word, name = "n")

dtm <- count_df %>%
  cast_dtm(document = doc_id, term = word, value = n)

# 7. dtm을 희소행렬로 변환 (Coherence 계산용)
dtm_sparse <- Matrix(as.matrix(dtm), sparse = TRUE)

# 8. 토픽 수(k) 그리드 서치: Perplexity vs Coherence\kk_list      <- 2:15
perplexities <- numeric(length(k_list))
coherences   <- numeric(length(k_list))

for (j in seq_along(k_list)) {
  k <- k_list[j]
  lda_fit <- LDA(dtm, k = k, method = "VEM", control = list(seed = 42))
  # Perplexity
  perplexities[j] <- perplexity(lda_fit, newdata = dtm)
  # Coherence
  post            <- topicmodels::posterior(lda_fit)
  phi_mat         <- post$terms
  coh_vec         <- CalcProbCoherence(phi = phi_mat, dtm = dtm_sparse, M = 10)
  coherences[j]   <- mean(coh_vec)
  message(sprintf("k=%2d: Perplexity=%.1f, Coherence=%.4f", k, perplexities[j], coherences[j]))
}

eval_df <- tibble(k = k_list, Perplexity = perplexities, Coherence = coherences)
print(eval_df)

# 9. 최적 토픽 수 선택 (예: Coherence 최대)
best_k <- eval_df$k[which.max(eval_df$Coherence)]
message("최적 토픽 수: ", best_k)

# 10. 최종 LDA 모델 학습
final_lda <- LDA(dtm, k = 12, method = "VEM", control = list(seed = 42))
final_post <- topicmodels::posterior(final_lda)

# 11. 토픽별 상위 단어 출력
library(tidytext)
topics <- tidy(final_lda, matrix = "beta")
top_terms <- topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  arrange(topic, -beta)
print(top_terms)

top_terms

# 12. 문서별 주요 토픽 할당
# final_post$topics 는 document × topic 확률 행렬

# 12. 토픽 레이블링: 사람이 이해하기 쉬운 주제 이름 할당
# 토픽 번호별 레이블 정의 (예시)
topic_labels <- c(
  '1'  = '친구·취향·관계',
  '2'  = '감정·스타일·외모',
  '3'  = '소셜미디어·놀이',
  '4'  = '미래·감정표현',
  '5'  = '이벤트·감정',
  '6'  = '호감·소통',
  '7'  = '성격·이미지',
  '8'  = '이동·자아탐색',
  '9'  = '호감·첫인상',
  '10' = '일상·감성',
  '11' = '인기·취미·언어',
  '12' = '여행·관심·컬러'
)

# 레이블을 top_terms 또는 doc_topics에 추가
top_terms <- top_terms %>%
  mutate(label = topic_labels[as.character(topic)])
print(top_terms)


# 13. 문서별 주요 토픽 할당
doc_topics <- as.data.frame(final_post$topics)
colnames(doc_topics) <- paste0("Topic", seq_len(ncol(doc_topics)))
doc_topics <- doc_topics %>%
  mutate(
    doc_id     = seq_len(nrow(doc_topics)),
    main_topic = max.col(select(., starts_with("Topic")))
  ) %>%
  mutate(topic_label = topic_labels[as.character(main_topic)])
print(head(doc_topics))

# 13. 문서별 주요 토픽 할당
# final_post$topics 는 document × topic 확률 행렬
doc_topics <- as.data.frame(final_post$topics)
colnames(doc_topics) <- paste0("Topic", seq_len(ncol(doc_topics)))
doc_topics <- doc_topics %>%
  mutate(
    doc_id     = seq_len(nrow(doc_topics)),
    main_topic = max.col(select(., starts_with("Topic")))
  ) %>%
  mutate(topic_label = topic_labels[as.character(main_topic)])

library(tidyr)

# 13.1. 누락된 문서(doc_id)에 대한 처리: 토큰이 없어 dtm에서 제외된 문서 포함
all_docs <- tibble(doc_id = seq_len(nrow(df)))
doc_topics_full <- all_docs %>%
  left_join(doc_topics, by = "doc_id") %>%
  mutate(
    main_topic   = replace_na(main_topic, 0L),
    topic_label  = replace_na(topic_label, "NoTopic")
  )
print(head(doc_topics_full))

# 14. 평가 플롯 (Perplexity vs Coherence) 평가 플롯 (Perplexity vs Coherence)
scale_factor <- max(eval_df$Perplexity) / max(eval_df$Coherence)

ggplot(eval_df, aes(x = k)) +
  geom_line(aes(y = Perplexity), size = 1) +
  geom_point(aes(y = Perplexity), size = 2) +
  geom_line(aes(y = Coherence * scale_factor), linetype = "dashed", size = 1) +
  geom_point(aes(y = Coherence * scale_factor), shape = 17, size = 2) +
  scale_y_continuous(
    name = "Perplexity",
    sec.axis = sec_axis(~ . / scale_factor, name = "Coherence")
  ) +
  labs(x = "토픽 수 (k)", title = "Perplexity vs Coherence") +
  theme_minimal()


# 문서 100번이 어떤 토픽에 속하는지
doc_topics %>% filter(doc_id == 100)

doc_topics %>%
  count(topic_label) %>%
  arrange(desc(n))

df %>%
  # 1) doc_id 생성 (질문 순서와 일치하도록)
  mutate(doc_id = row_number()) %>%
  # 2) 토픽 할당 정보 병합 (누락된 문서는 topic_label = "NoTopic" 처리된 doc_topics_full 사용)
  left_join(doc_topics_full %>% select(doc_id, topic_label), by = "doc_id") %>%
  # 3) 관심 있는 토픽으로 필터링
  filter(topic_label == "호감·첫인상") %>%
  # 4) 원본 질문 텍스트만 추출
  pull(question_text)



