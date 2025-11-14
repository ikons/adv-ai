# Bayesian Networks (Student Network)

Αυτός ο φάκελος περιέχει ένα μικρό παράδειγμα **Bayesian Network**
με το κλασικό μοντέλο "*Student*".

Το δίκτυο αυτό είναι ιδανικό για εισαγωγή σε:

- αναπαράσταση γνώσης με κατευθυνόμενους ακύκλους γράφους (DAGs),
- παραγοντοποίηση κοινής κατανομής ως γινόμενο conditional πιθανοτήτων,
- inference με χρήση αλγορίθμων όπως το Variable Elimination.


## 🔗 Μεταβλητές του δικτύου

- `Difficulty` ∈ {`easy`, `hard`}
- `Intelligence` ∈ {`low`, `high`}
- `Grade` ∈ {`A`, `B`, `C`}
- `SAT` ∈ {`low`, `high`}
- `Letter` ∈ {`weak`, `strong`}

Δομή (τόξα) του γράφου:

- `Difficulty → Grade`
- `Intelligence → Grade`
- `Intelligence → SAT`
- `Grade → Letter`


## 🎯 Διδακτικοί στόχοι

Με αυτό το παράδειγμα, οι φοιτητές:

- βλέπουν πώς ορίζουμε τη δομή ενός Bayesian Network με `pgmpy`,
- μαθαίνουν να ορίζουν πίνακες πιθανοτήτων (CPDs) με `TabularCPD`,
- ελέγχουν την ορθότητα του μοντέλου με `check_model()`,
- εκτελούν βασικά ερωτήματα inference:

  - $P(\text{Grade} \mid \text{Difficulty}, \text{Intelligence})$,
  - $P(\text{Intelligence} \mid \text{Grade}, \text{SAT})$,
  - $P(\text{Intelligence} \mid \text{Letter} = \text{strong})$ (explaining away).


## 📁 Περιεχόμενα φακέλου

- `train_student_bn.py`  
  Ορίζει τη δομή του Student Bayesian Network, τους CPDs
  και αποθηκεύει το μοντέλο σε αρχείο `.joblib`.

- `infer_student_bn.py`  
  Φορτώνει το αποθηκευμένο Bayesian Network και εκτελεί
  μερικά ενδεικτικά ερωτήματα inference με `VariableElimination`.

- `models/`  
  Περιέχει το αποθηκευμένο μοντέλο `student_bn.joblib`.


## ⚙️ Εγκατάσταση απαιτούμενων πακέτων

```bash
pip install pgmpy joblib matplotlib
```


## 🚀 Εκτέλεση

Από τη ρίζα του repository:

```bash
# 1️⃣ Δημιουργία και αποθήκευση του Bayesian Network
python -m bayesian_networks.train_student_bn

# 2️⃣ Εκτέλεση ερωτημάτων inference
python -m bayesian_networks.infer_student_bn
```


## 📓 Σχετικό notebook

Στον φάκελο `notebooks/` υπάρχει το notebook:

- `04_bayesian_networks_intro.ipynb`

το οποίο:

- επαναλαμβάνει τη θεωρία των Bayesian Networks,
- ορίζει το ίδιο Student network βήμα-βήμα,
- εκτελεί πολλά παραδείγματα inference,
- παράγει διαγράμματα (π.χ. bar plots πιθανότητας $P(\text{Intelligence} \mid \text{evidence})$)
  ώστε οι φοιτητές να βλέπουν εποπτικά πώς αλλάζουν οι posterior πιθανότητες.


## 🔁 Σύνδεση με το μάθημα

Το παράδειγμα αυτό μπορεί να χρησιμοποιηθεί:

- ως βασικό σημείο εισαγωγής στα Bayesian Networks,
- ως πρόγευση για πιο σύνθετα μοντέλα (π.χ. dynamic Bayesian networks),
- ως υποδομή για μικρές ασκήσεις (π.χ. αλλαγή CPDs, προσθήκη κόμβων,
  σύγκριση διαφορετικών σεναρίων evidence).

Δεν απαιτείται κάποιο εξωτερικό dataset: όλα τα στοιχεία της κατανομής
ορίζονται "στο χέρι" μέσω των CPDs του δικτύου.
