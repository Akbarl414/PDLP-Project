#ifndef PDLPClass_hpp
#define PDLPClass_hpp

#include <stdio.h>
#include <vector>
using namespace std;


class PDLP{
    public:
    //Public functions:
        // A Method to assign LP values
        void assignLpValues(int &num_row, int &num_col, int &num_nonZero, vector<double> &cost,
                            vector<double> &bound, vector<double> &lp_matrix_values, vector<int> &lp_matrix_index, 
                            vector<int> &lp_matrix_start,  string modelName, string filePathName);
        
        //Model initialization function: computes all internal data such as primal weight, bounds and costs norms and sets feasiblility tolerances
        void initialiseModel();

        //various methods to run the model including methods to run scaling algorithms
        void runPDHG(bool outputFlag = 1);
        void runFeasiblePDHG(bool outputFlag = 1, bool debugFlag = 0);
        void ruiz_Rescale();
        void run_Rescale();
        void chamPock_Rescale();

        //Output Functions to give information about the model after it has converged
        void printObjectiveValue(); 
        void printFullResults();
        void writeFile(double &runTime);

    //Public Objects
        //Model information such as bounds, costs, and constraint matrix data
        vector<double> costs;
        vector<double> bounds;
        int num_rows;
        int num_cols; 
        int num_nonZeros;
        vector<double> matrix_values;
        vector<int> matrix_index;
        vector<int> matrix_start;
       
        //Relevant information for the update information
        double step_size;
        double primal_step_size;
        double dual_step_size;
        double matrix_norm;
        vector<double> x_k;
        vector<double> x_k1;
        vector<double> y_k;
        vector<double> y_k1;

        //Result information to be outputed for analysis
        vector<double> x_k_result;
        vector<double> y_k_result;
        
        //Scaling vectors and scaled info for the constraint matrix
        vector<double> d_c;
        vector<double> d_r;
        vector<double> diag_c;
        vector<double> diag_r;
        vector<double> scaled_matrix_values;
        
        //Rescaling setup information: iteration cap and statuses
        double rrescale_iter_cap = 10; //Ruiz rescaling iteration cap
        bool statusRescale = 0; 
        bool chamPockStatus;
        bool alternate_Scaling = false;

        //Solver information such as debug status and model information
        string model;
        string filePath;
        int iter_cap;
        bool debugFlag = 0;
       
        //Step size flattening status 
        bool flatten_step_size = false;
    

    private:
    //Private functions:
        //Functions to initialize data for the problem
        void initialiseNorms(); //Initialize the bounds, costs, and matrix norms
        double matrixNorm(); //A value to compute the 2-norm of a non-square matrix

        //Functions for running the update and checking feasibilities
        void PDHGUpdate();
        bool isFeasible();
        bool isPrimalFeasible();
        bool isDualFeasible();
        bool isDualityGap();
        bool isComplementarity();
        void calculateReducedCosts(); //calculate the reduced costs (used in feasibility calculations)
        void restartSolve(); //To reset the primal and dual k, k+1 values

        //Functions for running the scaling methods and scaling the LP
        void scaleLP();
        void unScaleLP();
        void chamPock_Rescale_alternate();
        void ruiz_Rescale_alternate();

        //Helper functions for row and column scaling
        void scale_Column(vector<double> &scalingVector);
        void scale_Row(vector<double> &scalingVector);

        //Helper functions for Linear algebra
        vector<double> vectorSubtraction(vector<double> &vect1, vector<double> &vect2);
        double vectorEuclidianNorm(vector<double> &vect);
        void inverse_sqrt(vector<double> &vect);
        double primal_weight_norm(vector<double> &x, vector<double> &x_hash, vector<double> &y, vector<double> &y_hash );

        //Functions used in the updating of the method
        void getObjectiveValue();
        void printDebug();
        bool updateCriteria();
        void addResults();
        
        //An implementation of the adaptive step size enhancement
        double adaptive_step();
        
    //Private objects:
        //Model running inforation to keep track of the run
        bool up; 
        int iterations;

        //Feasibility information
        double feasibility_tolerance;
        double primal_feasibility_tolerance;
        double dual_feasibility_tolerance;
        double primal_relative_error; 
        double primal_2_norm; // ||Ax-b||_2
        double complementarity;
        double adjusted_complementarity;
        double dualityGap;

        //Values for feasibiltiy and printing information
        double objectiveValue;
        vector<double> reducedCosts;
        
        //Norms associated with variables
        double bounds_2_norm;
        double costs_2_norm;
        double dual_inf_norm;
        bool rescale_status; 
     
        //step size adjustment variables
        double step_balance;
    
        //Varaibles associated with scaling the LP    
        vector<double> post_scaled_x_k;
        vector<double> post_scaled_y_k;
        vector<double> orignal_costs;
        vector<double> orignal_bounds;
        vector<double> orignal_matrix_values;
        vector<double> col_norm;
        vector<double> row_norm;
        
};

#endif                                                                                                     
/* PDLPClass_hpp */