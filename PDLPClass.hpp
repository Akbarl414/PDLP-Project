#ifndef PDLPClass_hpp
#define PDLPClass_hpp

#include <stdio.h>
#include <vector>
using namespace std;




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
    vector<double> x_k_result;
    vector<double> y_k_result;
    vector<double> d_c;
    vector<double> d_r;
    vector<double> diag_c;
    vector<double> diag_r;
    vector<double> scaled_matrix_values;
    int iter_cap;
    double primal_step_size;
    double dual_step_size;
    bool debugFlag = 0;
    bool statusRescale = 0; 
    bool chamPockStatus;
    string model;
    string filePath;
    double rrescale_iter_cap = 10; 
    bool flatten_step_size = false;
    bool alternate_Scaling = false;




    void assignLpValues(int &num_row, int &num_col, int &num_nonZero, vector<double> &cost,
                        vector<double> &bound, vector<double> &lp_matrix_values, vector<int> &lp_matrix_index, 
                        vector<int> &lp_matrix_start,  string modelName, string filePathName);
    void initialiseModel();
    void runPDHG(bool outputFlag = 1);
    void runFeasiblePDHG(bool outputFlag = 1, bool debugFlag = 0);
    void printObjectiveValue();
    void printFullResults();
    void ruiz_Rescale();
    void run_Rescale();
    void chamPock_Rescale();
    void writeFile(double &runTime);



    private:
    bool up; 
    int iterations;
    double objectiveValue;
    vector<double> reducedCosts;
    double feasibility_tolerance;
    double primal_feasibility_tolerance;
    double dual_feasibility_tolerance;
    double primal_relative_error;
    double primal_2_norm;
    double complementarity;
    double adjusted_complementarity;
    double bounds_2_norm;
    double costs_2_norm;
    double dual_inf_norm;
    bool rescale_status; 
    double dualityGap;
    double step_balance;
    

    
    vector<double> post_scaled_x_k;
    vector<double> post_scaled_y_k;
    vector<double> orignal_costs;
    vector<double> orignal_bounds;
    vector<double> orignal_matrix_values;
    vector<double> col_norm;
    vector<double> row_norm;



    
    void initialiseNorms();
    double matrixNorm();
    void PDHGUpdate();
    vector<double> vectorSubtraction(vector<double> &vect1, vector<double> &vect2);
    double vectorEuclidianNorm(vector<double> &vect);
    bool updateCriteria();
    void getObjectiveValue();
    void restartSolve();
    void calculateReducedCosts();
    bool isFeasible();
    bool isPrimalFeasible();
    bool isDualFeasible();
    bool isDualityGap();
    bool isComplementarity();
    void printDebug();
    void inverse_sqrt(vector<double> &vect);
    void scaleLP();
    void unScaleLP();
    void chamPock_Rescale_alternate();
    void scale_Column(vector<double> &scalingVector);
    void scale_Row(vector<double> &scalingVector);
    void addResults();
    void ruiz_Rescale_alternate();
    double adaptive_step();
    double primal_weight_norm(vector<double> &x, vector<double> &x_hash, vector<double> &y, vector<double> &y_hash );
};

#endif                                                                                                     
/* PDLPClass_hpp */