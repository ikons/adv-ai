# Bayesian Learning & Naive Bayes (Αφελής μπεϋζιανή πρόβλεψη) (SMS Spam & Iris)  

Αυτός ο φάκελος περιέχει εκπαιδευτικά παραδείγματα **Bayesian learning**
με χρήση διαφορετικών παραλλαγών του Naive Bayes:

1. **Gaussian Naive Bayes** σε πρόβλημα ταξινόμησης με συνεχείς μεταβλητές (Iris dataset).
2. **Multinomial Naive Bayes** σε πρόβλημα ταξινόμησης κειμένου (SMS Spam).


## 1. Gaussian Naive Bayes – Iris dataset

Εδώ θέλουμε να δείξουμε την παραλλαγή **Gaussian Naive Bayes**, η οποία
είναι κατάλληλη για continuous χαρακτηριστικά.

Χρησιμοποιούμε το κλασικό **Iris dataset** (ενσωματωμένο στη scikit-learn),
που περιέχει συνεχείς μετρήσεις λουλουδιών:

- `sepal length`,
- `sepal width`,
- `petal length`,
- `petal width`,

και τρεις κλάσεις:

- `setosa`,
- `versicolor`,
- `virginica`.

Η υπόθεση του Gaussian NB είναι ότι κάθε χαρακτηριστικό $x_i$ ακολουθεί
κανονική κατανομή μέσα σε κάθε κλάση $y = k$:

$P(x_i \mid y = k) = \mathcal{N}(x_i \mid \mu_{k,i}, \sigma^2_{k,i})$

οπότε (με την υπόθεση ανεξαρτησίας):

$P(y \mid x) \propto P(y) \prod_i \mathcal{N}(x_i \mid \mu_{y,i}, \sigma^2_{y,i}).$


### 1.1 Αρχεία κώδικα (Iris / Gaussian NB)

- `train_gaussian_nb_iris.py`  
  Φορτώνει το Iris dataset από τη scikit-learn, κάνει `train_test_split`,
  εφαρμόζει προαιρετικά `StandardScaler` και εκπαιδεύει `GaussianNB`.
  Εκτυπώνει `classification_report`, αποθηκεύει:
  - το μοντέλο σε `models/gaussian_nb_iris.joblib`,
  - τον scaler σε `models/gaussian_nb_iris_scaler.joblib`,
  - ένα διάγραμμα confusion matrix σε `models/gaussian_nb_iris_cm.png`.

- `infer_gaussian_nb_iris.py`  
  Φορτώνει scaler + μοντέλο και υπολογίζει προβλέψεις για
  ενδεικτικά διανύσματα χαρακτηριστικών (4-διάστατα vectors),
  εκτυπώνοντας την προβλεπόμενη κλάση και τις posterior πιθανότητες.


### 1.2 Δεδομένα

Δεν απαιτείται εξωτερικό CSV, καθώς το Iris dataset φορτώνεται απευθείας
μέσω:

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data      # συνεχείς μετρήσεις
y = iris.target    # κλάσεις (0, 1, 2)
```


### 1.3 Εκτέλεση (Iris / Gaussian NB)

Από τη ρίζα του repository:

```bash
# Εκπαίδευση Gaussian NB στο Iris dataset
python -m bayesian_learning.train_gaussian_nb_iris

# Inference σε ενδεικτικά παραδείγματα
python -m bayesian_learning.infer_gaussian_nb_iris
```

Σχετικό notebook:

- `notebooks/03a_gaussian_nb_iris.ipynb`

Το notebook:

- περιέχει σύντομη θεωρία Gaussian Naive Bayes,
- φορτώνει το Iris dataset και παρουσιάζει τα βασικά χαρακτηριστικά,
- εκπαιδεύει Gaussian NB και εμφανίζει confusion matrix,
- απεικονίζει πώς αλλάζουν οι posterior πιθανότητες για διαφορετικά δείγματα,
- ενθαρρύνει πειραματισμούς με διαφορετικά splits ή χωρίς κανονικοποίηση.



## 2. Multinomial Naive Bayes – SMS Spam

Το σενάριο είναι κλασικό: ταξινομούμε SMS μηνύματα σε δύο κατηγορίες:

- *ham* (κανονικά μηνύματα)
- *spam* (ανεπιθύμητα / διαφημιστικά)

Χρησιμοποιούμε το μοντέλο **Multinomial Naive Bayes**, το οποίο είναι
κατάλληλο για δεδομένα καταμέτρησης (counts), όπως Bag-of-Words ή TF-IDF.

Η βασική ιδέα:

- Χρησιμοποιούμε τον τύπο του Bayes
  $P(y \mid x) \propto P(y) \prod_i P(x_i \mid y)$,
- όπου τα $x_i$ είναι οι συχνότητες/βαθμοί των λέξεων στο μήνυμα,
- και η "αφελής" (naive) υπόθεση είναι ότι τα $x_i$ είναι conditionally independent (υπό συνθήκη ανεξάρτητα) δεδομένης της κλάσης $y$.



### 2.1 Θεωρία NLP: Bag-of-Words, Λεξιλόγιο & TF-IDF  

Η επεξεργασία φυσικής γλώσσας (NLP) απαιτεί να μετατρέπουμε κείμενο σε αριθμούς,  
ώστε να μπορεί να χρησιμοποιηθεί από αλγόριθμους μηχανικής μάθησης.  
Οι πιο βασικές και θεμελιώδεις τεχνικές είναι:

- **Bag-of-Words (BoW)**
- **TF-IDF (Term Frequency – Inverse Document Frequency)**

Παρακάτω αναλύονται όλες οι έννοιες αναλυτικά.

---

### 2.1.1 Δημιουργία Λεξιλογίου (Vocabulary)

Το πρώτο βήμα στην αναπαράσταση κειμένου είναι η κατασκευή ενός **λεξιλογίου**.

#### Τι είναι το λεξιλόγιο;

Είναι μια λίστα με όλες τις *μοναδικές* λέξεις που εμφανίζονται στα μηνύματα του dataset.

Παράδειγμα:

Αν έχουμε τα SMS:

1. “free winner call now”  
2. “hey, are we still on for coffee tomorrow?”  
3. “call now for free prize!”

Τότε οι μοναδικές λέξεις είναι:

```text
["free", "winner", "call", "now", "hey", "are", "we", "still", "on",
 "for", "coffee", "tomorrow", "prize"]
```

Αυτό είναι το **λεξιλόγιο (vocabulary)**.

#### Πώς κατασκευάζεται;

1. Παίρνουμε όλα τα SMS.  
2. Κάνουμε **tokenization** (σπάμε κάθε μήνυμα σε λέξεις).  
3. Κατεβάζουμε όλα σε πεζά.  
4. Αφαιρούμε σημεία στίξης.  
5. Προαιρετικά αφαιρούμε stopwords (π.χ. “the”, “and”, “is”).  
6. Κρατάμε μόνο τις ΜΟΝΑΔΙΚΕΣ λέξεις.  
7. Ταξινομούμε το λεξιλόγιο (προαιρετικά).

Αυτό ορίζει τον «χώρο λέξεων» στον οποίο θα γίνει η αναπαράσταση.

---

### 2.1.2 Bag-of-Words (BoW)

Η μέθοδος Bag-of-Words μετατρέπει κάθε μήνυμα σε ένα **διάνυσμα αριθμών**,  
όπου κάθε θέση αντιστοιχεί σε μια λέξη του λεξιλογίου και η τιμή δείχνει  
πόσες φορές εμφανίζεται η λέξη στο μήνυμα.

#### Παράδειγμα

Λεξιλόγιο:

```text
["free", "winner", "call", "hey", "coffee"]
```

SMS:  
**“free free winner call now”**

BoW vector:

```text
[2, 1, 1, 0, 0]
```

Ανάλυση:

- “free” → 2 φορές  
- “winner” → 1 φορά  
- “call” → 1 φορά  
- “hey” → 0 φορές  
- “coffee” → 0 φορές  

Το BoW είναι **πολύ απλό**, αλλά έχει δύο μειονεκτήματα:

- Δεν λαμβάνει υπόψη τη σειρά των λέξεων.  
- Οι συχνές λέξεις σε όλα τα μηνύματα (π.χ. “call”, “now”) δεν είναι πολύ χρήσιμες.

Για αυτό περνάμε στο TF-IDF.

---

### 2.1.3 TF-IDF (Term Frequency – Inverse Document Frequency)

Το TF-IDF είναι μια βελτιωμένη μορφή του BoW.  
Δίνει μεγαλύτερη βαρύτητα σε λέξεις που:

- εμφανίζονται συχνά στο συγκεκριμένο μήνυμα (**TF**),
- αλλά είναι σχετικά σπάνιες σε όλα τα μηνύματα (**IDF**).

Η ιδέα είναι ότι λέξεις όπως “free”, “winner”, “prize” (τυπικές σε spam)
πρέπει να έχουν μεγαλύτερο βάρος από λέξεις όπως “call”, “now”, που
εμφανίζονται παντού.

---

### 2.1.4 TF — Term Frequency (ίδιο με τα counts του BoW)

Ο **TF** μετράει πόσες φορές εμφανίζεται μια λέξη στο συγκεκριμένο μήνυμα.

Αν συμβολίσουμε με $TF_{i,j}$ τη συχνότητα της λέξης $i$ στο μήνυμα $j$, τότε:

$$
TF_{i,j} = \text{count of word i in message j}
$$

Στο ίδιο παράδειγμα:

- μήνυμα: “free free winner call now”  
- λεξιλόγιο: `["free", "winner", "call", "hey", "coffee"]`

έχουμε:

```text
TF(free)   = 2
TF(winner) = 1
TF(call)   = 1
TF(hey)    = 0
TF(coffee) = 0
```

Άρα το **TF διάνυσμα** είναι ακριβώς το ίδιο με το BoW διάνυσμα:

```text
[2, 1, 1, 0, 0]
```

Δηλαδή μέχρι εδώ, **δεν αλλάζει η μορφή του vector**, μόνο το όνομα που του δίνουμε.

---

### 2.1.5 IDF — Inverse Document Frequency (σπανιότητα λέξης)

Ο **IDF** μετράει πόσο σπάνια είναι μια λέξη στο σύνολο των μηνυμάτων.

Αν $N$ είναι ο συνολικός αριθμός των μηνυμάτων και $df_i$ είναι ο αριθμός
των μηνυμάτων που περιέχουν τη λέξη $i$, τότε:

$$
IDF_i = \log \frac{N}{df_i}
$$

- Σπάνια λέξη (μικρό $df_i$) → μεγάλο $IDF_i$  
- Πολύ συχνή λέξη (μεγάλο $df_i$) → μικρό $IDF_i$

#### Παράδειγμα με τα 3 SMS από πριν

1. “free winner call now”  
2. “hey, are we still on for coffee tomorrow?”  
3. “call now for free prize!”

Με λεξιλόγιο `["free", "winner", "call", "hey", "coffee"]`:

- “free” εμφανίζεται στα SMS 1 και 3 → $df_	ext{free} = 2$  
- “winner” εμφανίζεται μόνο στο SMS 1 → $df_	ext{winner} = 1$  
- “call” εμφανίζεται στα SMS 1 και 3 → $df_	ext{call} = 2$  
- “hey” εμφανίζεται μόνο στο SMS 2 → $df_	ext{hey} = 1$  
- “coffee” εμφανίζεται μόνο στο SMS 2 → $df_	ext{coffee} = 1$  

Αν $N = 3$, τότε, ενδεικτικά:

- $IDF_\text{free} = \log \frac{3}{2}$  
- $IDF_\text{winner} = \log 3$  
- $IDF_\text{call} = \log \frac{3}{2}$  
- $IDF_\text{hey} = \log 3$  
- $IDF_\text{coffee} = \log 3$  

Μπορούμε να το σκεφτούμε ως **διάνυσμα IDF** για το λεξιλόγιο:

```text
[IDF(free), IDF(winner), IDF(call), IDF(hey), IDF(coffee)]
= [log(3/2), log(3), log(3/2), log(3), log(3)]
```

---

### 2.1.6 TF-IDF = TF × IDF (ίδιο διάνυσμα, άλλες τιμές)

Τώρα συνδυάζουμε TF και IDF:

$$
TFIDF_{i,j} = TF_{i,j} \cdot IDF_i
$$

Δηλαδή, για κάθε θέση του διανύσματος:

- κρατάμε το **ίδιο λεξιλόγιο**,  
- κρατάμε το **ίδιο μήκος διανύσματος**,  
- αλλά αντικαθιστούμε τα counts με **weights TF×IDF**.

Στο παράδειγμά μας, το TF διάνυσμα ήταν:

```text
[2, 1, 1, 0, 0]
```

και το IDF διάνυσμα:

```text
[IDF(free), IDF(winner), IDF(call), IDF(hey), IDF(coffee)]
= [log(3/2), log(3), log(3/2), log(3), log(3)]
```

Άρα το **TF-IDF διάνυσμα** για το ίδιο SMS και το ίδιο λεξιλόγιο είναι:

```text
[ 2 * IDF(free),
  1 * IDF(winner),
  1 * IDF(call),
  0 * IDF(hey),
  0 * IDF(coffee) ]
```

ή αλλιώς, πιο συνοπτικά:

```text
[ 2·log(3/2), 1·log(3), 1·log(3/2), 0, 0 ]
```

Συνοψίζοντας:

- Στην **BoW / TF** μορφή το διάνυσμα είναι:  

  ```text
  [2, 1, 1, 0, 0]
  ```

- Στην **TF-IDF** μορφή (για το ίδιο SMS, το ίδιο vocabulary) γίνεται:  

  ```text
  [2·IDF_free, 1·IDF_winner, 1·IDF_call, 0, 0]
  ```

Έτσι:

- η λέξη “winner” (σπάνια και τυπική σε spam) αποκτά **μεγάλο βάρος**,  
- η λέξη “call” (πολύ συχνή σε πολλά SMS) αποκτά **σχετικά μικρότερο βάρος**.

Αυτό το TF-IDF διάνυσμα είναι ακριβώς αυτό που τροφοδοτούμε στο
πολυωνυμικό μοντέλο Naive Bayes.

---

### 2.1.7 Παραδείγματα TF-IDF σε ham / spam

#### SMS (ham)

**“Hey, are we still on for coffee tomorrow?”**

| λέξη     | TF-IDF (ενδεικτικό) |
|----------|---------------------|
| hey      | 0.41                |
| coffee   | 0.55                |
| tomorrow | 0.37                |
| free     | 0.00                |

#### SMS (spam)

**“WINNER!! You have won a 1000$ cash prize. Call now!”**

| λέξη   | TF-IDF (ενδεικτικό) |
|--------|---------------------|
| winner | 0.83                |
| prize  | 0.79                |
| cash   | 0.76                |
| call   | 0.30                |


### 2.2 Αρχεία κώδικα (SMS Spam)

- `train_naive_bayes_sms.py`  
  Εκπαίδευση πολυωνυμικού (multinomial) μοντέλου Naive Bayes για ταξινόμηση SMS.
  Χρησιμοποιεί `TfidfVectorizer` + `MultinomialNB` μέσα σε `Pipeline`,
  εκτυπώνει `classification_report` και αποθηκεύει:
  - το εκπαιδευμένο pipeline σε `.joblib`,
  - ένα διάγραμμα confusion matrix σε `.png`.

- `infer_naive_bayes_sms.py`  
  Φορτώνει το αποθηκευμένο μοντέλο και κάνει inference (πρόβλεψη) σε νέα SMS,
  εκτυπώνοντας την προβλεπόμενη κλάση (*ham* / *spam*) και τις
  posterior (εκ των υστέρων) πιθανότητες $P(\text{ham} \mid x)$, $P(\text{spam} \mid x)$.

- `models/`  
  Περιέχει αρχεία `.joblib` και σχετικά διαγράμματα.


### 2.3 Δεδομένα

Τα scripts περιμένουν το αρχείο:

- `data/sms_spam.csv`

με τις στήλες:

- `label` — τιμή `ham` ή `spam`
- `text`  — κείμενο SMS (string)

Παράδειγμα:

```text
label,text
ham,"Hey, are we still on for coffee tomorrow?"
spam,"WINNER!! You have won a 1000$ cash prize. Call now!"
```

Δες το κεντρικό `data/README.md` για οδηγίες λήψης από Kaggle
και μετατροπή του *SMS Spam Collection* στη μορφή `sms_spam.csv`.


### 2.4 Εκτέλεση (SMS Spam)

Από τη ρίζα του repository:

```bash
# Εκπαίδευση μοντέλου
python -m bayesian_learning.train_naive_bayes_sms     --alpha 1.0     --max_features 10000     --test_size 0.2     --random_state 0

# Inference σε ενδεικτικά μηνύματα
python -m bayesian_learning.infer_naive_bayes_sms
```




Σχετικό notebook:

- `notebooks/03b_multinomial_naive_bayesian_learning.ipynb`



Το notebook:

- περιέχει σύντομη θεωρία Bayes & Naive Bayes,
- φορτώνει το `sms_spam.csv`,
- εκπαιδεύει Multinomial NB,
- παράγει διαγράμματα (class balance, confusion matrix, bar plot με top λέξεις),
- δείχνει posterior πιθανότητες για νέα μηνύματα.

