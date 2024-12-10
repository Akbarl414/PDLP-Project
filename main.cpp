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

    int test = 0; 
    for(int i =0; i<10; i++){
        cout << test << "\t";
        test += max(test, 10);
    }
    printf("\n test value is %i \n", test);

    h.setOptionValue("output_flag", false);
    h.getStandardFormLp(num_cols, num_rows, num_nonZeros, offsets, c.data(),
                        b.data(), A_start.data(), A_index.data(),
                        A_value.data());

    printf("%i non zeros, num row = %i and num col = %i \n", num_nonZeros, num_rows, num_cols);
    
    cout<< "PDLP class test \n";
    PDLP model1;
    model1.assignLpValues(num_rows, num_cols, num_nonZeros, c, b, A_value, A_index, A_start);
    if(argc > 2 && stod(argv[2]) != 0) {
        s = stod(argv[2]);
        model1.step_size = s;   
    }
    debugFlag = 0;
    if(argc > 3) debugFlag = 1;
    printf("Argc is %i long \n", argc);
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

    if(debugFlag)
    {
        for(int iCol =0; iCol < num_cols; iCol++)
        {
        if (cost[iCol] != 0 && solution.col_value[iCol] != model1.x_k[iCol])
            printf("HiGHS gets %g while PDLP gets %g which is a %g difference \n",solution.col_value[iCol], model1.x_k[iCol], abs(solution.col_value[iCol] - model1.x_k[iCol]));
        }

    }
    
    // for (HighsInt iCol = 0; iCol < num_col; iCol++)
    //     printf("Col %1d has optimal primal value %g; dual value %g\n", int(iCol), solution.col_value[iCol], solution.col_dual[iCol]);
    // for (HighsInt iRow = 0; iRow < num_row; iRow++)
    //     printf("Row %1d has optimal primal value %g; dual value %g\n", int(iRow), solution.row_value[iRow], solution.row_dual[iRow]);
}
   

  
