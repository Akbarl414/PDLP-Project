#ifndef PDLPClass_hpp
#define PDLPClass_hpp

#include <stdio.h>
#include <vector>
using namespace std;


class Rectangle
{
    public: 
    double width; 
    double length;
    void initWhy(bool &value);
    void printDimensions();
    void printWhy();


    private:
    bool why;
    void changeWhy();

};

class PDLP
{
    public:
    int num_rows;
    int num_cols; 
    int num_nonZeros;
    double step_size;
    vector<double> costs;
    vector<double> bounds;
    vector<double> matrix_values;
    vector<int> matrix_index;
    vector<int> matrix_start;
    double matrix_norm;
    vector<double> x_k;
    vector<double> x_k1;
    vector<double> y_k;
    vector<double> y_k1;

    void assignLpValues(int &num_row, int &num_col, int &num_nonZero, vector<double> &cost,
    vector<double> &bound, vector<double> &lp_matrix_values, vector<int> &lp_matrix_index, 
    vector<int> &lp_matrix_start);
    void runPDHG(bool outputFlag = 1);
    void printObjectiveValue();
    void printFullResults();
    


    private:
    bool up; 
    int iterations;
    double objectiveValue;

    double matrixNorm();
    void PDHGUpdate();
    vector<double> vectorSubtraction(vector<double> &vect1, vector<double> &vect2);
    double vectorEuclidianNorm(vector<double> &vect);
    bool updateCriteria();
    void getObjectiveValue();
    void restartSolve();

};

#endif                                                                                                     
/* PDLPClass_hpp */