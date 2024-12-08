
import pickle as pk
import numpy as np

# Load the _accs.pkl file from OUTPUT_DIR
loaded = pk.load(open('OUTPUT_DIR/evaluation_results_accs.pkl', 'rb'))


# Compute the mean accuracy across folds
mean_subj_acc_across_folds = loaded.mean(0)

print("Mean accuracy across folds:", mean_subj_acc_across_folds)


np.savetxt("mean_accuracies.csv", mean_subj_acc_across_folds, delimiter=",")
print("Mean accuracies saved to mean_accuracies.csv")


import matplotlib.pyplot as plt

plt.plot(mean_subj_acc_across_folds)
plt.title("Mean Accuracy Across Folds")
plt.xlabel("Voxel/Feature Index")
plt.ylabel("Mean Accuracy")
plt.show()


top_indices = mean_subj_acc_across_folds.argsort()[-10:][::-1]  # Top 10 accuracies
print("Top 10 Features:", top_indices)
print("Top 10 Accuracies:", mean_subj_acc_across_folds[top_indices])
