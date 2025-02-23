//
//  main.cpp
//  MathsProjectCode
//
//  Created by Akbar Latif on 10/18/24.
//

#include <iostream>
#include "ProjectFunctions.hpp"
#include "PDLPClass.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include "Highs.h"

using namespace std;

int main(int argc, const char *argv[])
{
    /****************************
    Declare all relevant variables
    *****************************/
    string model;
    int num_rows;
    int num_cols;
    int num_nonZeros;
    double offsets;
    Highs h;
    double s;
    int max_iterations;
    bool debugFlag;
    /**********************************
    Assign the relevant model, use AVGAS as default instance
    ************************************/
    if (argc < 2)
    {
        model = "avgas";
    }
    else
    {
        model = argv[1];
    }
    string model_file = "/Users/akbarlatif/Desktop/scriptz/ExampleLPs/" + model + ".mps";
    
    /**********************************
    Get all the relevant information from the model into our function
    ************************************/
    h.readModel(model_file);
    h.setOptionValue("output_flag", false);
    h.getStandardFormLp(num_cols, num_rows, num_nonZeros, offsets);

    //Initialise more variables that will be used later
    vector<double> c(num_cols);
    vector<double> b(num_rows);
    vector<double> A_value(num_nonZeros);
    vector<int> A_index(num_nonZeros);
    vector<int> A_start(num_cols + 1);

   
    h.setOptionValue("output_flag", false);
    h.getStandardFormLp(num_cols, num_rows, num_nonZeros, offsets, c.data(),
                        b.data(), A_start.data(), A_index.data(),
                        A_value.data());


    /*******************************************
      It's debugging matricies time:
    *******************************************/
    
    //Running a test on Ruiz scaling for the matrix that I have written on my iPad that follows the form described in matrix t
    // vector<double> t_values = {2,4,3,7,4,3,2,2,6,2,1,1,6,8};    
    // vector<int> t_index = {0,1,2,3,0,1,2,3,0,1,3,1,2,3};
    // vector<int> t_start = {0,4,8,11,12,14};
    // num_rows = 4;
    // num_cols = 5;
    // num_nonZeros = 14;

    // vector<double> p_values = {1,1,2420,1.58};
    // vector<int> p_index = {0,1,0,1};
    // vector<int> p_start = {0,2,4};    
    // num_rows = 2;
    // num_cols = 2;
    // num_nonZeros = 4;
    

    // PDLP rTest;
    // //Sense check that the algorithm works with a non-square matrix:
    //rTest.assignLpValues(num_rows, num_cols, num_nonZeros, c, b, t_values, t_index, t_start);
    //rTest.assignLpValues(num_rows, num_cols, num_nonZeros, c, b, p_values, p_index, p_start);


    // //Now a test with the matrix values of a model's matrix
    // rTest.assignLpValues(num_rows, num_cols, num_nonZeros, c, b, A_value, A_index, A_start);
        
    // rTest.ruiz_Rescale();


    /*******************************************
      End the debugging
    *******************************************/

   /*******************************************
      Run my PDLP version currently commented out until finished debug
    *******************************************/

    // printf("%i non zeros, num row = %i and num col = %i \n", num_nonZeros, num_rows, num_cols);
    bool chamPock = 0; 
    printf("PDLP class test on %s \n", model.c_str());
    PDLP model1;
    model1.assignLpValues(num_rows, num_cols, num_nonZeros, c, b, A_value, A_index, A_start);
    debugFlag = 0;
    const char *debug = "debug";
    const char *rescale = "rescale";
    const char *input4 = "";  
    const char *input3 = argv[3];
    if(argc > 5 && stod(argv[5]) != 0) {
      max_iterations = stod(argv[5]);
      model1.iter_cap = max_iterations;   
    }
    // printf("argv 3 is %s and the debug const char is %s \n", argv[3], debug);
    if(strcmp(debug, input3) == 0) debugFlag = 1;
    model1.debugFlag = debugFlag; 
    if(argc > 6 && stod(argv[6]) != 0) {
      chamPock = stoi(argv[6]);
      model1.chamPockStatus = chamPock;   
    }
    if(argc > 4){
        input4 = argv[4];
        if(strcmp(rescale, input4) == 0){
        model1.statusRescale = 1; 
        printf("Running Rescaling \n");
        model1.run_Rescale();
        } 
    } 
    //This initialiseModel() function is made to make the ||A||_2 norm be in terms of the scaled matrix, but I'm not sure if it works 
    model1.initialiseModel();
    if(argc > 2 && stod(argv[2]) != 0) {
        s = stod(argv[2]);
        model1.step_size = s;   
    }
    model1.runFeasiblePDHG(1, debugFlag);
    //model1.runPDHG();
    model1.printObjectiveValue();
    cout << "end of test \n";


    /*************************************************
    Using HiGHS to solve the current model as our sense check
    ***********************************************/
    HighsSparseMatrix m;
    HighsStatus status;
    status = h.readModel(model_file);
    assert(status == HighsStatus::kOk);
    h.setOptionValue("output_flag", false);
    h.run();

    double original_objective_function_value = h.getInfo().objective_function_value;

    HighsInt num_col;
    HighsInt num_row;
    HighsInt num_nz;
    double offset;

    status = h.getStandardFormLp(num_col, num_row, num_nz, offset);
    assert(status == HighsStatus::kOk);

    std::vector<double> cost(num_col);
    std::vector<double> rhs(num_row);
    std::vector<HighsInt> start(num_col + 1);
    std::vector<HighsInt> index(num_nz);
    std::vector<double> value(num_nz);
    h.getStandardFormLp(num_col, num_row, num_nz, offset, cost.data(),
                        rhs.data(), start.data(), index.data(),
                        value.data());

    HighsLp standard_form_lp;
    printf("Num of HiGHS rows %i, Num of HiGHS cols %i \n", num_row, num_col);
    standard_form_lp.num_col_ = num_col;
    standard_form_lp.num_row_ = num_row;
    standard_form_lp.offset_ = offset;
    standard_form_lp.col_cost_ = cost;
    standard_form_lp.col_lower_.assign(num_col, 0);
    standard_form_lp.col_upper_.assign(num_col, kHighsInf);
    standard_form_lp.row_lower_ = rhs;
    standard_form_lp.row_upper_ = rhs;
    standard_form_lp.a_matrix_.start_ = start;
    standard_form_lp.a_matrix_.index_ = index;
    standard_form_lp.a_matrix_.value_ = value;
    h.passModel(standard_form_lp);
    double objective_function_value = h.getInfo().objective_function_value;

    double rel_objective_function_value_diff = std::abs(original_objective_function_value - objective_function_value) /
                                               std::max(1.0, std::abs(original_objective_function_value));

    printf("For Ml,OG,StdFm,dl,%s,%g,%g,%g\n", model.c_str(),
           original_objective_function_value,
           objective_function_value,
           rel_objective_function_value_diff);
    // assert(rel_objective_function_value_diff < 1e-10);
    h.run();
    HighsSolution solution = h.getSolution();

    // if(debugFlag)
    // {
    //     for(int iCol =0; iCol < num_cols; iCol++)
    //     {
    //     if (cost[iCol] != 0 && solution.col_value[iCol] != model1.x_k[iCol])
    //         printf("HiGHS gets %g while PDLP gets %g which is a %g difference \n",solution.col_value[iCol], model1.x_k[iCol], abs(solution.col_value[iCol] - model1.x_k[iCol]));
    //     }

    // }
    
    // for (HighsInt iCol = 0; iCol < num_col; iCol++)
    //     printf("Col %1d has optimal primal value %g; dual value %g\n", int(iCol), solution.col_value[iCol], solution.col_dual[iCol]);
    // for (HighsInt iRow = 0; iRow < num_row; iRow++)
    //     printf("Row %1d has optimal primal value %g; dual value %g\n", int(iRow), solution.row_value[iRow], solution.row_dual[iRow]);
}
   

  
