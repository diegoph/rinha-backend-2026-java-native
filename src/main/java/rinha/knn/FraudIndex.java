package rinha.knn;

public interface FraudIndex {
    int size();
    int searchFraudCount(short[] q);
}
