import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregando o arquivo ARFF usando a biblioteca scipy
data = arff.loadarff("spambase.arff")
df = pd.DataFrame(data[0])

# Convertendo o tipo de dados da variável de destino para discreto
df['class'] = df['class'].astype(int)

# Dividindo os dados em atributos (X) e classe (y)
X = df.drop('class', axis=1)
y = df['class']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando modelos de árvore de decisão e Naive Bayes
tree_model = DecisionTreeClassifier()
naive_bayes_model = GaussianNB()

# Treinando os modelos
tree_model.fit(X_train, y_train)
naive_bayes_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste para cada modelo
tree_y_pred = tree_model.predict(X_test)
naive_bayes_y_pred = naive_bayes_model.predict(X_test)

# Calculando a taxa de acerto para cada modelo
tree_accuracy = accuracy_score(y_test, tree_y_pred)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_y_pred)

# Calculando as instâncias classificadas corretamente e incorretamente para cada modelo
tree_correct_instances = sum(tree_y_pred == y_test)
naive_bayes_correct_instances = sum(naive_bayes_y_pred == y_test)
tree_incorrect_instances = sum(tree_y_pred != y_test)
naive_bayes_incorrect_instances = sum(naive_bayes_y_pred != y_test)

# Obtendo o número total de instâncias e atributos
num_instances = df.shape[0]
num_attributes = df.shape[1] - 1  # Exclui a coluna da variável de destino

# Calculando a quantidade de e-mails de spam e não spam
num_spam_emails = sum(y == 1)
num_nonspam_emails = sum(y == 0)

# Calculando a porcentagem de e-mails de spam e não spam
tree_total_instances = tree_correct_instances + tree_incorrect_instances
bayes_total_instances = naive_bayes_correct_instances + naive_bayes_incorrect_instances
spam_percentage = (num_spam_emails / num_instances) * 100
nonspam_percentage = (num_nonspam_emails / num_instances) * 100

# Imprimindo os resultados para cada modelo
print("Resultado para Árvore de Decisão:")
print("Quantidade de Instâncias: {}".format(num_instances))
print("Quantidade de Atributos: {}".format(num_attributes))
print("Quantidade de E-mails de Spam: {}".format(num_spam_emails))
print("Porcentagem de E-mails de Spam: {:.2f}%".format(spam_percentage))
print("Quantidade de E-mails Não Spam: {}".format(num_nonspam_emails))
print("Porcentagem de E-mails Não Spam: {:.2f}%".format(nonspam_percentage))
print("Instâncias classificadas corretamente: {}".format(tree_correct_instances))
print("Instâncias classificadas incorretamente: {}".format(tree_incorrect_instances))
print("Total Number of Instances: {}".format(tree_total_instances))
print("Taxa de acerto: {:.2f}%".format(tree_accuracy * 100))

print("\nResultado para Naive Bayes:")
print("Quantidade de Instâncias: {}".format(num_instances))
print("Quantidade de Atributos: {}".format(num_attributes))
print("Quantidade de E-mails de Spam: {}".format(num_spam_emails))
print("Porcentagem de E-mails de Spam: {:.2f}%".format(spam_percentage))
print("Quantidade de E-mails Não Spam: {}".format(num_nonspam_emails))
print("Porcentagem de E-mails Não Spam: {:.2f}%".format(nonspam_percentage))
print("Instâncias classificadas corretamente: {}".format(naive_bayes_correct_instances))
print("Instâncias classificadas incorretamente: {}".format(naive_bayes_incorrect_instances))
print("Total Number of Instances: {}".format(bayes_total_instances))
print("Taxa de acerto: {:.2f}%".format(naive_bayes_accuracy * 100))

# Comparando os algoritmos com validação cruzada
models = [('Árvore de Decisão', tree_model), 
          ('Naive Bayes', naive_bayes_model)]

for model_name, model in models:
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\nValidação Cruzada para {}: ".format(model_name))
    print("Taxas de acerto em cada fold: ", cv_scores)
    print("Taxa de acerto média: {:.2f}%".format(cv_scores.mean() * 100))
