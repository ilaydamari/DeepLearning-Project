# 🎵 Lyrics Generation with Deep Learning 

## תיאור הפרויקט
פרויקט זה מממש מודל למידה עמוקה ליצירת מילות שירים באמצעות ארכיטקטורת RNN (Recurrent Neural Networks). המודל מאומן על מאגר נתונים של מילות שירים ויכול ליצור מילים חדשות בהתבסס על טקסט התחלתי שניתן על ידי המשתמש.

## 🏗️ ארכיטקטורת הפרויקט

```
assignment3/
├── 📄 train.py                    # סקריפט האימון הראשי
├── 📄 generate.py                 # סקריפט לגנרציה (עתיד להתפתח)
├── 📁 data/
│   └── sets/
│       ├── lyrics_train_set.csv   # נתוני האימון
│       └── lyrics_test_set.csv    # נתוני הבדיקה
├── 📁 models/
│   ├── RNN_baseline.py            # מודל RNN הבסיסי
│   ├── RNN_baseline_V1.py         # גרסאות נוספות של המודל
│   └── RNN_baseline_V2.py
├── 📁 utils/
│   ├── text_utils.py              # כלים לעיבוד טקסט
│   └── midi_features.py           # כלים לעיבוד MIDI (עתיד)
└── 📁 embeddings/                 # תיקיית embeddings
```

## 🔬 הפייפליין של Data Science

### 1. **Data Loading & Preprocessing Pipeline** 📊
**קובץ**: `utils/text_utils.py` + `train.py`

#### שלבים:
1. **`parse_lyrics_csv()`** - טוען מילות שירים מקבצי CSV
   - קורא נתונים מהטבלאות
   - מנקה מאופיינים לא רצויים ('&', ',,,,')
   - מסנן מילות שירים קצרות מדי

2. **`TextPreprocessor.clean_text()`** - ניקוי טקסט
   - הופך טקסט לאותיות קטנות
   - מסיר סימני פיסוק
   - מנרמל רווחים
   - מבצע טוקניזציה בסיסית

3. **`TextPreprocessor.build_vocabulary()`** - בניית מילון מילים
   - ספירת תדירות מילים
   - יצירת מיפוי `word2idx` ו-`idx2word`
   - הוספת טוקנים מיוחדים: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
   - סינון מילים לפי תדירות מינימלית

### 2. **Word Embeddings Pipeline** 🔤
**קובץ**: `utils/text_utils.py`

#### שלבים:
1. **`load_word2vec_embeddings()`** - טוען מודל Word2Vec מוכן
   - משתמש במודל Google News 300D
   - יוצר מטריצת embeddings
   
2. **`_create_embedding_matrix()`** - בניית מטריצת embeddings
   - מטריצה בגודל `[vocab_size, 300]`
   - אתחול עם Word2Vec עבור מילים קיימות
   - אתחול אקראי עבור מילים חדשות
   - וקטור אפסים עבור PAD token

### 3. **Sequence Preparation Pipeline** 📝
**קובץ**: `utils/text_utils.py`

#### שלבים:
1. **`text_to_sequence()`** - המרת טקסט לרצף מספרים
   - הוספת SOS token בהתחלה
   - המרת מילים לאינדקסים
   - החלפת מילים לא מוכרות ב-UNK token
   - הוספת EOS token בסוף

2. **`prepare_sequences()`** - הכנת רצפים לאימון
   - יצירת sliding window sequences
   - כל רצף הוא Input X ו-Target Y (המילה הבאה)
   - Padding לאורך אחיד
   - מטריצות numpy מוכנות לPyTorch

### 4. **Model Architecture Pipeline** 🧠
**קובץ**: `models/RNN_baseline.py`

#### ארכיטקטורת המודל:
```python
class LyricsRNN(nn.Module):
    def __init__(self):
        # 1. Embedding Layer (300D Word2Vec)
        self.embedding = nn.Embedding(vocab_size, 300)
        
        # 2. RNN Layer (LSTM/GRU)
        self.rnn = nn.LSTM(300, hidden_size, num_layers, dropout=0.3)
        
        # 3. Dropout Layer
        self.dropout_layer = nn.Dropout(0.3)
        
        # 4. Output Projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)
```

#### זרימת המידע:
1. **Input**: רצף אינדקסים `[batch_size, seq_len]`
2. **Embedding**: `[batch_size, seq_len, 300]`
3. **RNN**: `[batch_size, seq_len, hidden_size]`
4. **Output**: `[batch_size, seq_len, vocab_size]`

### 5. **Training Pipeline** 🏋️‍♂️
**קובץ**: `train.py` + `models/RNN_baseline.py`

#### LyricsRNNTrainer - מחלקת האימון:

**שלבי האימון**:
1. **`train_step()`** - צעד אימון יחיד
   - Forward pass
   - חישוב Loss (CrossEntropyLoss)
   - Backward propagation
   - Gradient clipping (max_norm=1.0)
   - Update weights

2. **`validate_step()`** - צעד validation
   - Forward pass ללא gradients
   - חישוב validation loss

#### Training Loop בפונקציה `train_model()`:
```python
for epoch in range(num_epochs):
    # Training Phase
    for batch in train_loader:
        loss = trainer.train_step(input_batch, target_batch)
        
    # Validation Phase  
    for batch in val_loader:
        val_loss = trainer.validate_step(input_batch, target_batch)
        
    # Learning Rate Scheduling
    scheduler.step(avg_val_loss)
    
    # Early Stopping Check
    if val_loss < best_val_loss:
        save_best_model()
    else:
        patience_counter += 1
```

### 6. **Text Generation Pipeline** ✨
**קובץ**: `models/RNN_baseline.py`

#### פונקציית `generate_text()`:
1. **Initialization**: טוען seed sequence
2. **Autoregressive Generation**:
   ```python
   for _ in range(max_length):
       # Forward pass
       output_logits = model(current_sequence)
       
       # Temperature scaling
       logits = logits / temperature
       
       # Top-k sampling
       top_k_logits, indices = torch.topk(logits, k)
       
       # Sample next word
       next_word = torch.multinomial(probabilities, 1)
       
       # Append to sequence
       generated_sequence.append(next_word)
   ```

### 7. **Evaluation Pipeline** 📊
**קובץ**: `train.py`

#### מדדי הערכה:
1. **Loss**: CrossEntropyLoss על test set
2. **Perplexity**: `exp(loss)` - מדד לאי וודאות המודל
3. **Generated Text Quality**: בדיקה איכותית של טקסט שנוצר

## 📈 מטריקות ומדדים

### Loss Function
```python
criterion = nn.CrossEntropyLoss(ignore_index=0)  # מתעלם מ-PAD tokens
```

### Perplexity Calculation
```python
perplexity = np.exp(cross_entropy_loss)
```
- פרפלקסיטי נמוכה = מודל טוב יותר
- פרפלקסיטי של ~100-200 נחשבת טובה לגנרציית טקסט

### Learning Rate Scheduling
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
```

## 🎛️ היפר-פרמטרים

```python
config = {
    'max_sequence_length': 50,    # אורך רצף מקסימלי
    'batch_size': 32,             # גודל batch
    'embedding_dim': 300,         # ממד Word2Vec
    'hidden_size': 512,           # גודל hidden state
    'num_layers': 2,              # מספר שכבות RNN
    'dropout': 0.3,               # קצב dropout
    'learning_rate': 0.001,       # קצב למידה
    'min_word_freq': 2,           # תדירות מילה מינימלית
}
```

## 🚀 הרצת הפרויקט

### דרישות מקדימות
```bash
pip install torch torchvision pandas numpy matplotlib seaborn gensim tqdm
```

### הרצת אימון
```bash
python train.py
```

### תהליך האימון יכלול:
1. ✅ טעינת נתונים ועיבוד טקסט
2. ✅ בניית מילון מילים
3. ✅ טעינת Word2Vec embeddings  
4. ✅ אימון המודל עם early stopping
5. ✅ הערכה על test set
6. ✅ גנרציית דוגמאות טקסט
7. ✅ שמירת מודל ומטריקות

## 📁 פלטים וקבצים
- `models/best_lyrics_model.pth` - המודל המאומן הטוב ביותר
- `models/preprocessor.pkl` - הpreprocessor השמור
- `training_curves.png` - גרפים של loss ו-perplexity
- Console output עם מדדים ודוגמאות טקסט

## 🎵 דוגמאות גנרציה

המודל יכול ליצור מילות שיר בהתבסס על טקסט התחלתי:

**Input**: "love is"  
**Output**: "love is a beautiful thing that makes me feel alive..."

**Input**: "in the night"  
**Output**: "in the night when stars are shining bright..."

## 🔧 הרחבות עתידיות
- [ ] מודל Transformer למילות שירים
- [ ] אינטגרציה עם MIDI features
- [ ] ממשק אינטראקטיבי לגנרציה
- [ ] מדדי הערכה איכותיים נוספים

## 📚 מקורות והשראה
- ארכיטקטורת RNN מהקורס Deep Learning  
- Word2Vec embeddings מ-Google News
- טכניקות text generation עדכניות

---
**פרויקט במסגרת**: הנדסת נתונים - למידה עמוקה, סמסטר ז'