# 📋 מדריך הכנת דוח Assignment 3 - Melody-Conditioned Lyrics Generation

## 🎯 מבנה הדוח הנדרש

### חלק א': רקע תיאורטי
### חלק ב': מתודולוגיה ויישום
### חלק ג': ניסויים ותוצאות
### חלק ד': ניתוח והשוואות
### חלק ה': מסקנות

---

## 🚀 שלב 1: הכנות ראשוניות

### 1.1 בדיקת מבנה הפרויקט
```powershell
# ריצה בטרמינל
ls -la
```
**מה להעתיק לדוח:**
```
Directory structure:
├── train.py
├── train_melody.py  
├── generate_melody.py
├── evaluation.py
├── models/
├── data/
└── utils/
```

### 1.2 בדיקת נתונים
```powershell
python -c "
import pandas as pd
import os

print('=== DATA OVERVIEW ===')
train_df = pd.read_csv('data/sets/lyrics_train_set.csv')
print(f'Training songs: {len(train_df)}')

midi_train = len([f for f in os.listdir('data/midi/train') if f.endswith('.mid')])
midi_test = len([f for f in os.listdir('data/midi/test') if f.endswith('.mid')])
print(f'MIDI train files: {midi_train}')
print(f'MIDI test files: {midi_test}')
"
```

---

## 🚀 שלב 2: אימון המודל הבסיסי

### 2.1 אימון RNN בסיסי
```powershell
python train.py --model_type lstm --hidden_size 128 --num_layers 2 --epochs 50 --batch_size 32
```

**מה לתעד:**
- Training loss curve
- Validation perplexity  
- זמן אימון
- דוגמאות גנרציה

**מה להעתיק מהפלט:**
```
Final Results:
Training Loss: X.XXX
Validation Perplexity: XX.XX
Training Time: XX minutes
Sample Generation: "..."
```

### 2.2 אימון GRU להשוואה
```powershell
python train.py --model_type gru --hidden_size 128 --num_layers 2 --epochs 50 --batch_size 32
```

---

## 🚀 שלב 3: אימון מודלים מותני-מלודיה

### 3.1 גישה A - Concatenation Approach
```powershell
python train_melody.py --model_type melody_concat --hidden_size 128 --num_layers 2 --epochs 50 --batch_size 16
```

**מה לתעד:**
```
=== MELODY CONCATENATION MODEL ===
Architecture: Word(300D) + Melody(84D) → 384D → LSTM → Output
Training Loss: X.XXX
Validation Perplexity: XX.XX
Melody Alignment Score: X.XXX
```

### 3.2 גישה B - Conditioning Approach  
```powershell
python train_melody.py --model_type melody_condition --hidden_size 128 --num_layers 2 --epochs 50 --batch_size 16
```

**מה לתעד:**
```
=== MELODY CONDITIONING MODEL ===
Architecture: Melody(84D) → Hidden Init + Word(300D) → LSTM → Output
Training Loss: X.XXX
Validation Perplexity: XX.XX
Conditioning Effectiveness: X.XXX
```

---

## 🚀 שלב 4: גנרציה והערכה מקיפה

### 4.1 גנרציה בסיסית
```powershell
python generate_melody.py --model_path models/best_baseline_model.pth --model_type baseline --interactive
```

**דוגמאות לבדיקה:**
- "love is"
- "in the night" 
- "music makes me"
- "dancing to the"

**מה להעתיק:**
```
Input: "love is"
Generated: "love is a beautiful song that makes my heart sing with joy and happiness"

Input: "in the night"  
Generated: "in the night when stars shine bright above the city lights"
```

### 4.2 גנרציה מותנית-מלודיה
```powershell
python generate_melody.py --model_path models/best_melody_concat_model.pth --model_type melody_concat --midi_file data/midi/test/example.mid --seed_words "love heart"
```

**מה לתעד:**
```
MIDI File: example.mid
Musical Features: 
  - Key: C Major
  - Tempo: 120 BPM  
  - Rhythm Complexity: 0.75

Generated Lyrics:
[Verse]
Love heart beating like a drum tonight
Feel the music flowing through my soul
Every note brings me closer to the light  
Dancing to the rhythm makes me whole
```

### 4.3 הערכה מקיפה
```powershell
python quick_eval.py
```

**מה להעתיק מהפלט:**
```
=== PROJECT EVALUATION SUMMARY ===

📊 Model Performance:
├── Baseline LSTM: Perplexity 45.2, Diversity 0.73
├── Melody Concat: Perplexity 42.1, Diversity 0.78, Melody Align 0.85
└── Melody Condition: Perplexity 43.5, Diversity 0.76, Melody Align 0.82

🎵 Generation Quality:
├── Lyrical Coherence: 0.78
├── Creativity Score: 0.72  
├── Structure Quality: 0.81
└── Rhyme Quality: 0.69

✅ Assignment Requirements: 20/22 completed
```

### 4.4 הערכה מפורטת
```powershell
python evaluation.py --comprehensive --output_dir results/
```

---

## 🚀 שלב 5: ניתוח השוואתי

### 5.1 השוואת מודלים
```powershell
python generate_melody.py --compare_models --test_midi_dir data/midi/test/ --output_dir comparison_results/
```

**טבלת השוואה לדוח:**
```
| Model Type | Perplexity | Diversity | Melody Alignment | Training Time |
|------------|------------|-----------|------------------|---------------|
| Baseline LSTM | 45.2 | 0.73 | N/A | 25 min |
| Melody Concat | 42.1 | 0.78 | 0.85 | 45 min |  
| Melody Condition | 43.5 | 0.76 | 0.82 | 40 min |
```

### 5.2 ניתוח איכותי
```powershell
python generate_melody.py --qualitative_analysis --midi_file data/midi/test/upbeat_song.mid --seed_words "happy dance"
```

**דוגמאות השוואה לדוח:**

**בסיסי:**
"happy dance music makes me feel good today"

**מותנה במלודיה:**
```
[Verse]
Happy dance beneath the shining lights
Moving to the rhythm of the beat  
Feel the joy that lifts me to new heights
Music makes my heart skip to the heat
```

---

## 📝 מבנה הדוח המפורט

### חלק א': רקע תיאורטי (2-3 עמודים)
```
1. הקדמה - בעיית המחקר
2. סקירה ביבליוגרפית - מודלים קיימים
3. התרומה החדשנית - melody conditioning
4. מטרות המחקר
```

### חלק ב': מתודולוגיה (3-4 עמודים)
```
1. ארכיטקטורת המודל
   - Baseline RNN
   - Approach A: Concatenation  
   - Approach B: Conditioning

2. עיבוד נתונים
   - Text preprocessing
   - MIDI feature extraction (84D)
   - Temporal alignment

3. פרטי אימון
   - Loss functions
   - Optimization
   - Regularization
```

### חלק ג': ניסויים ותוצאות (4-5 עמודים)
```
1. הגדרות ניסוי
   - Dataset splits
   - Hyperparameters
   - Evaluation metrics

2. תוצאות כמותיות
   - טבלת השוואה (מהשלב 5.1)
   - גרפי learning curves
   - מטריקות הערכה

3. תוצאות איכותיות  
   - דוגמאות גנרציה
   - ניתוח מבנה שיריים
   - התאמה למלודיה
```

### חלק ד': ניתוח ודיון (2-3 עמודים)
```
1. השוואת גישות
   - יתרונות וחסרונות
   - מקרי קצה
   
2. הערכה ביקורתית
   - הגבלות המחקר  
   - אתגרים טכניים

3. התאמה לדרישות Assignment
   - כיסוי 22 הדרישות
```

### חלק ה': מסקנות (1-2 עמודים)
```
1. עיקרי הממצאים
2. תרומה מדעית
3. כיווני פיתוח עתידיים
4. יישומים פרקטיים
```

---

## 📊 קבצים לצירוף לדוח

### קבצי קוד עיקריים:
- `models/MelodyRNN.py` (ארכיטקטורת המודל)
- `train_melody.py` (פייפליין האימון)
- `evaluation.py` (מערכת ההערכה)

### תוצאות וגרפים:
- `training_curves.png` 
- `melody_alignment_analysis.png`
- `comparison_table.csv`
- `sample_generations.txt`

### דוחות אוטומטיים:
- `results/evaluation_report.html`
- `results/model_comparison.json`

---

## ⚡ טיפים לדוח מקצועי

### 1. עיצוב וקריאות
- השתמש בטבלאות מסודרות
- הוסף גרפים וויזואליזציות
- שמור על פורמט עקבי

### 2. תוכן איכותי  
- הסבר כל החלטה טכנית
- צרף דוגמאות קונקרטיות
- בצע השוואה לעבודות קיימות

### 3. ניתוח מעמיק
- הראה הבנה תיאורטית
- נתח כשלים ומגבלות
- הצע שיפורים עתידיים

---

## 🎯 רשימת בדיקה לדוח

### תוכן טכני ✅
- [ ] הסבר ארכיטקטורה מפורט
- [ ] תוצאות ניסויים מלאות  
- [ ] השוואה בין גישות
- [ ] ניתוח איכותי של הגנרציה

### דרישות פורמליות ✅  
- [ ] ביבליוגרפיה
- [ ] נומרציה של איורים/טבלאות
- [ ] מבנה לוגי וברור
- [ ] סיכום ומסקנות

### קבצים לצירוף ✅
- [ ] קוד מקור מלא
- [ ] תוצאות ניסויים
- [ ] דוגמאות גנרציה
- [ ] גרפים וויזואליזציות

---

**זמן מוערך לביצוע:** 4-6 שעות (2 שעות ריצות + 4 שעות כתיבה)
**אורך דוח מומלץ:** 12-15 עמודים + נספחים