import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv(r"D:\bgx\Pemrosesan Teks\Scraping_CaseFolding\Labeling\labeled_data.csv")   # ubah sesuai nama file Anda
df['label'] = df['label'].astype('object')
df_labeled = df[df['label'].notnull()]
df_unlabeled = df[df['label'].isnull()]
print("Jumlah labeled:", len(df_labeled))
print("Jumlah unlabeled:", len(df_unlabeled))


# ==========================================================
# 2. TF-IDF
# ==========================================================
vectorizer = TfidfVectorizer()
X_labeled = vectorizer.fit_transform(df_labeled['full_text'])
y_labeled = df_labeled['label']
X_unlabeled = vectorizer.transform(df_unlabeled['full_text'])


# ==========================================================
# 3. SELF TRAINING SETUP
# ==========================================================
base_model = LogisticRegression(max_iter=200)

self_training = SelfTrainingClassifier(
    base_model,
    threshold=0.8,
    verbose=True
)


# ==========================================================
# 4. GABUNGKAN LABELED + UNLABELED
# ==========================================================
y_unlabeled = np.array([-1] * len(df_unlabeled))
X_combined = np.vstack((X_labeled.toarray(), X_unlabeled.toarray()))
y_combined = np.concatenate((y_labeled, y_unlabeled))


# ==========================================================
# 5. TRAIN SELF TRAINING
# ==========================================================
self_training.fit(X_combined, y_combined)
print("\nSelf-training selesai!\n")


# ==========================================================
# 6. AMBIL HASIL LABEL OTOMATIS
# ==========================================================
pseudo_labels = self_training.predict(X_unlabeled)
df_unlabeled['label'] = pseudo_labels
df_final = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

# simpan hasil
df_final.to_csv("hasil_self_training.csv", index=False)
print("File hasil disimpan sebagai hasil_self_training.csv")
