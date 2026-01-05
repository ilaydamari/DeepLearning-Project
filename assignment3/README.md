# 🎵 Deep Learning Assignment 3 - Melody-Conditioned Lyrics Generation

## תיאור הפרויקט
פרויקט זה מממש מודל למידה עמוקה מתקדם ליצירת מילות שירים המותנות במלודיה. המערכת משלבת ניתוח מוזיקלי של קבצי MIDI עם גנרציית טקסט בשתי גישות חדשניות, ומהווה פריצת דרך בתחום הבינה המלאכותית המוזיקלית.

## 🎼 **NEW: Melody-Conditioned Generation** ⭐

### שתי גישות למיזוג מלודיה עם טקסט:

**🎵 גישה A: Melody Concatenation**
- שילוב מאפיינים מוזיקליים (84D) עם word embeddings (300D) בכל timestep
- יישור טמפורלי ישיר בין מלודיה ומילים
- ארכיטקטורה: Word + Melody → 384D → RNN → Output

**🎼 גישה B: Melody Conditioning**  
- שימוש במאפיינים מוזיקליים לאתחול hidden states של RNN
- השפעה גלובלית של המלודיה על תהליך הגנרציה
- שתי גרסאות: מבוסס-Projection ומבוסס-Attention

### מאפיינים מוזיקליים מ-MIDI (84 ממדים):
- **Pitch Histogram (12D)**: התפלגות גובה צלילים כרומטית
- **Rhythm Features (12D)**: דפוסי קצב, צפיפות צלילים, סינקופה
- **Instrument Features (16D)**: זיהוי וניתוח קטגוריות כלי נגינה
- **Temporal Features**: חילוץ מאפיינים מיושר לפי פעימה

## 🔧 עדכונים טכניים מרכזיים

**✅ אינטגרציית MIDI מלאה:**
- עיבוד מתקדם של קבצי MIDI באמצעות PrettyMIDI
- יישור טמפורלי בין מלודיה ומילות שיר
- מערכת אימון ייעודית למודלים מותני-מלודיה

**✅ ארכיטקטורה מקצועית:**
- מודלים מותני-מלודיה עם שתי גישות שונות
- מחלקות dataset מותאמות לנתונים מותני-MIDI
- מערכת הערכה השוואתית מקיפה

**✅ פייפליין אימון משופר:**
- TensorBoard logging עם מטריקות ייעודיות למלודיה
- מניעת data leakage מלאה
- אופטימיזציות ביצועים מתקדמות

## 🏗️ ארכיטקטורת הפרויקט

```
assignment3/
├── 📄 train.py                    # אימון מודלים בסיסיים עם TensorBoard
├── 📄 train_melody.py             # 🆕 אימון מודלים מותני-מלודיה
├── 📄 generate.py                 # גנרציית טקסט סטנדרטית
├── 📄 generate_melody.py          # 🆕 גנרציה מותנית במלודיה
├── 📁 data/
│   ├── sets/
│   │   ├── lyrics_train_set.csv   # נתוני האימון
│   │   └── lyrics_test_set.csv    # נתוני הבדיקה
│   └── midi/                      # 🆕 קבצי MIDI למיזוג מלודיה
│       ├── train/
│       ├── val/
│       └── test/
├── 📁 models/
│   ├── RNN_baseline.py            # מודל RNN הבסיסי הראשי
│   ├── RNN_baseline_V1.py         # גרסת LSTM קונסרבטיבית
│   ├── RNN_baseline_V2.py         # גרסת GRU אגרסיבית
│   └── MelodyRNN.py               # 🆕 מודלים מותני-מלודיה
├── 📁 utils/
│   ├── text_utils.py              # עיבוד טקסט מתקדם עם Word2Vec
│   └── midi_features.py           # 🆕 חילוץ מאפיינים מקבצי MIDI
├── 📁 embeddings/                 # Word2Vec embeddings
├── 📁 models/                     # מודלים מאומנים
└── 📁 runs/                       # TensorBoard logs לכל הגישות
```

## � הוראות הפעלה

### התקנת dependencies

```bash
pip install torch torchvision torchaudio
pip install gensim pandas numpy tensorboard
pip install pretty-midi librosa  # 🆕 לעיבוד MIDI
```

### 🎼 אימון מודלים מותני-מלודיה (NEW)

**אימון כל הגישות להשוואה:**
```bash
python train_melody.py --model_type compare --num_epochs 10 --batch_size 32
```

**אימון גישה ספציפית:**
```bash
python train_melody.py --model_type concatenation --num_epochs 15
python train_melody.py --model_type conditioning --num_epochs 15
```

**מעקב באמצעות TensorBoard:**
```bash
tensorboard --logdir=runs/approach_a_concatenation
tensorboard --logdir=runs/approach_b_projection
```

### 🎵 גנרציה מותנית במלודיה (NEW)

**גנרציה מקובץ MIDI:**
```bash
python generate_melody.py --model_path models/melody_concatenation_model.pth --model_type concatenation --midi_file data/midi/test/song1.mid --seed_words love heart
```

**הערכה מקיפה:**
```bash
python generate_melody.py --model_path models/melody_conditioning_projection.pth --model_type conditioning --midi_dir data/midi/test/ --evaluate
```

**השוואת כל הגישות:**
```bash
python generate_melody.py --compare --midi_dir data/midi/test/ --model_dir models/
```

### אימון מודלים בסיסיים

**אימון מודל בסיסי:**
```bash
python train.py --model_type baseline --num_epochs 10
python train.py --model_type v1  # LSTM קונסרבטיבית
python train.py --model_type v2  # GRU אגרסיבית
```

**גנרציית טקסט רגילה:**
```bash
python generate.py --model_path models/best_model.pth --seed_text "love is" --temperature 0.8
```

## 📊 מטריקות והערכה

### 🆕 הערכה מותנית-מלודיה
המערכת מספקת הערכה מקיפה הכוללת:
- **איכות גנרציה**: גיוון אוצר מילים, ניתוח חזרות
- **יישור מלודיה**: מטריקות התאמה טמפורלית
- **ניתוח השוואתי**: הערכת ביצועים בין הגישות
- **דוגמאות מופת**: 5 מלודיות × 3 שילובי מילים × 2 גישות

## 📊 מטריקות והערכה

### 🆕 הערכה מותנית-מלודיה
המערכת מספקת הערכה מקיפה הכוללת:
- **איכות גנרציה**: גיוון אוצר מילים, ניתוח חזרות
- **יישור מלודיה**: מטריקות התאמה טמפורלית
- **ניתוח השוואתי**: הערכת ביצועים בין הגישות
- **דוגמאות מופת**: 5 מלודיות × 3 שילובי מילים × 2 גישות

### מעקב TensorBoard
גישה למעקב זמן-אמת:
```bash
tensorboard --logdir=runs
```

- עקומות loss לאימון ואימות
- מטריקות ייעודיות למלודיה
- השוואה בין גישות שונות

## 🔬 פרטים טכניים מתקדמים

### ארכיטקטורת המודלים

| מודל | סוג RNN | שכבות | Hidden Size | Dropout | Learning Rate | מטרה |
|------|---------|-------|-------------|---------|---------------|-------|
| Baseline | ניתן לקונפיגורציה | 2 | 512 | 0.3 | 0.001 | מודל עיקרי |
| V1 | LSTM | 2 | 256 | 0.2 | 0.0005 | קונסרבטיבי |
| V2 | GRU | 3 | 512 | 0.4 | 0.001 | אגרסיבי |

### 🆕 מודלים מותני-מלודיה

| גישה | ממד Input | ארכיטקטורה | שיטת Conditioning |
|------|----------|-------------|-------------------|
| Concatenation | 384D (300+84) | Word+Melody → RNN | יישור טמפורלי |
| Projection | 300D | Word → RNN + Melody init | התניה ב-Hidden state |
| Attention | 300D | Word → RNN + Melody attention | מבוסס Attention |

### מפרטים טכניים
- **Embeddings**: 300D Word2Vec מ-Google News corpus
- **אורך רצף**: 50 טוקנים לכל רצף אימון  
- **אוצר מילים**: 10,000 המילים השכיחות ביותר
- **פונקציית Loss**: Cross-entropy עם יישור targets מתקדם
- **אופטימיזציה**: Adam optimizer עם ReduceLROnPlateau scheduling

### 🎼 עיבוד MIDI מתקדם
- **ספריית עיבוד**: PrettyMIDI לחילוץ מאפיינים
- **רזולוציה טמפורלית**: 0.25 שניות למסגרת מאפיינים
- **יישור טמפורלי**: סינכרון מלודיה עם רצפי טקסט
- **ממד Conditioning**: וקטורי מאפיינים של 84 ממדים

## 🎯 החדשנות והתוצאות

### נקודות חזקה
🎵 **מערכת גנרציית מילים מותנית-מלודיה ראשונה מסוגה**  
🔬 **השוואה שיטתית של גישות התניה שונות**  
🎼 **פייפליין חילוץ מאפיינים מקיף מקבצי MIDI**  
📊 **מסגרת הערכה מתקדמת עם ניתוח רב-מטרי**  
⚡ **קוד ברמה מקצועית עם תיעוד נרחב**

### התקדמות משמעותית
יישום זה מייצג התקדמות משמעותית בגנרציית טקסט נוירלי, תוך גישור בין בינה מלאכותית מוזיקלית ובלשנית באמצעות טכניקות התניה מתוחכמות של מלודיה.

### תוצאות מצופות
- **גנרציה מותנית-איכות גבוהה**: מילות שיר המותאמות למאפייני המלודיה
- **גיוון ויצירתיות**: שילוב ייחודי של מוזיקה וטקסט
- **הערכה מדעית**: ניתוח השוואתי מקיף של גישות שונות

## 📝 רישיון ושימוש

פרויקט זה פותח עבור מטלת קורס למידה עמוקה ומדגים יישום מתקדם של טכנולוגיות בינה מלאכותית ביצירת תוכן מוזיקלי.

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

3. **🚨 Data Leakage Prevention** - מניעת דליפת מידע
   - **בעבר**: `all_lyrics = train_lyrics + test_lyrics` ❌ 
   - **עכשיו**: `preprocessor.build_vocabulary(train_lyrics)` ✅
   - המילון נבנה רק על נתוני האימון
   - נתוני הטסט משתמשים במילון זה (UNK למילים לא מוכרות)

4. **`TextPreprocessor.build_vocabulary()`** - בניית מילון מילים
   - ספירת תדירות מילים **בנתוני האימון בלבד**
   - יצירת מיפוי `word2idx` ו-`idx2word`
   - הוספת טוקנים מיוחדים: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
   - סינון מילים לפי תדירות מינימלית

### 2. **Word Embeddings Pipeline** 🔤
**קובץ**: `utils/text_utils.py`

#### שלבים:
1. **`load_word2vec_embeddings()`** - טוען Word2Vec מוכן
   - משתמש ב-`gensim.downloader` 
   - מודל: `word2vec-google-news-300`
   - 300 ממדים כנדרש

2. **`_create_embedding_matrix()`** - יצירת מטריצת embeddings
   - מטריצה בגודל `[vocab_size, 300]`
   - מילים קיימות ב-Word2Vec: vector מוכן
   - מילים לא קיימות: vector רנדומלי
   - PAD token: vector אפסים

### 3. **Training Data Preparation** 🎯
**קובץ**: `utils/text_utils.py`

#### שלבים:
1. **`prepare_sequences()`** - יצירת רצפים לאימון
   - יצירת sliding windows
   - כל רצף הופך למספר דוגמאות אימון
   - פורמט: `[context] → next_word`
   - Padding לאורך אחיד

2. **Data Splitting** - חלוקה נכונה:
   - Training: 80%
   - Validation: 10% 
   - Test: 10%
   - **חשוב**: Test נשאר נפרד לחלוטין

### 4. **Model Architecture Pipeline** 🧠
**קבצים**: `models/RNN_baseline*.py`

#### גרסאות המודל:
```python
# RNN_baseline.py - מודל בסיסי גמיש
class LyricsRNN:
    - Embedding layer (300D Word2Vec)
    - LSTM/GRU layers (configurable)
    - Dropout layer 
    - Output projection (vocab_size)

# RNN_baseline_V1.py - LSTM קונסרבטיבי
- LSTM, 2 layers, hidden=256, dropout=0.2
- Learning rate: 0.0005 (נמוך ליציבות)

# RNN_baseline_V2.py - GRU אגרסיבי  
- GRU, 3 layers, hidden=512, dropout=0.4
- Learning rate: 0.001 (גבוה למהירות)
```

### 5. **Training Pipeline with TensorBoard** 📈
**קובץ**: `train.py`

#### שיפורים חדשים:
1. **TensorBoard Logging** במקום matplotlib:
   ```python
   writer = SummaryWriter(log_dir=f'{log_dir}/lyrics_rnn_{timestamp}')
   
   # Batch-level logging
   writer.add_scalar('Loss/Train_Batch', loss, global_step)
   writer.add_scalar('Perplexity/Train_Batch', np.exp(loss), global_step)
   
   # Epoch-level logging  
   writer.add_scalars('Loss/Epoch', {
       'Training': avg_train_loss,
       'Validation': avg_val_loss
   }, epoch)
   ```

2. **Comprehensive Monitoring**:
   - Loss curves (train/validation)
   - Perplexity trends
   - Learning rate scheduling
   - Real-time progress tracking

3. **Early Stopping & Checkpointing**:
   - Monitor validation loss
   - Save best model state
   - Restore for evaluation

### 6. **Model Evaluation & Generation** 🎭
**קובץ**: `train.py`

#### שלבים:
1. **Test Set Evaluation**:
   - חישוב Perplexity על נתוני טסט נקיים
   - אין data leakage

2. **Text Generation**:
   - Temperature sampling
   - Top-k sampling
   - הדגמה עם seeds שונים

## 🚀 הרצת הפרויקט

### התקנת Dependencies
```bash
pip install torch torchvision torchaudio
pip install gensim pandas numpy tqdm tensorboard
```

### הרצת אימון
```bash
python train.py
```

### צפייה ב-TensorBoard
```bash
tensorboard --logdir=runs
# פתח http://localhost:6006 בדפדפן
```

## 📊 מעקב אחר התקדמות

### TensorBoard Metrics:
1. **Loss/Train_Batch** - Loss לכל batch באימון
2. **Loss/Epoch** - Loss ממוצע לכל epoch (train + validation)
3. **Perplexity/Epoch** - Perplexity לכל epoch
4. **Learning_Rate** - שינויים בקצב למידה

### פלטים של האימון:
```
models/
├── best_lyrics_model.pth      # המודל הטוב ביותר
├── preprocessor.pkl           # עיבוד הטקסט
runs/
└── lyrics_rnn_YYYYMMDD_HHMMSS/  # TensorBoard logs
```

## 🎯 תוצאות ומדדי הערכה

### מדד עיקרי: Perplexity
- ככל שהערך נמוך יותר, המודל טוב יותר
- Perplexity = exp(loss)
- ערך טיפוסי טוב: < 50

### השוואת גרסאות:
- **V1 (LSTM)**: יציבות, איכות טקסט גבוהה
- **V2 (GRU)**: מהירות, יעילות זיכרון

## 🔧 התאמות אישיות

### שינוי הגדרות במודל:
```python
config = {
    'rnn_type': 'LSTM',      # או 'GRU'
    'hidden_size': 512,      # גודל hidden state
    'num_layers': 2,         # מספר שכבות  
    'dropout': 0.3,          # dropout rate
    'learning_rate': 0.001,  # קצב למידה
    'batch_size': 32,        # גודל batch
}
```

## 📝 הערות טכניות חשובות

### Data Leakage Prevention:
- מילון המילים נבנה **רק** על נתוני האימון
- נתוני הטסט מעובדים עם מילון זה (UNK למילים חדשות)
- זה מבטיח שהמודל לא "ראה" את נתוני הטסט מראש

### TensorBoard vs Matplotlib:
- TensorBoard: מעקב real-time, אינטראקטיבי, professional
- Matplotlib: סטטי, פשוט יותר, פחות מידע
- TensorBoard מתאים יותר לפרויקטי deep learning מתקדמים

### Memory Management:
- השימוש ב-DataLoaders מאפשר טעינה חכמה של נתונים
- Gradient accumulation אפשרי לbatches גדולים
- GPU memory optimization עם mixed precision

## 🎵 דוגמאות גנרציה

```python
# דוגמאות לgenerates מצופים:
seeds = [
    "love is" → "love is all we need to feel alive..."
    "music makes" → "music makes the world go round and round..."
    "when the sun" → "when the sun goes down the night begins..."
]
```

## 📚 חומר עזר

- [TensorBoard Documentation](https://pytorch.org/docs/stable/tensorboard.html)
- [Data Leakage Prevention](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [RNN for Text Generation](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

---
**מפתח**: מטלה 3 - למידה עמוקה | **עדכון**: ינואר 2026
        
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