//
//  main.cpp
//  MathsProjectCode
//
//  Created by Akbar Latif on 10/18/24.
//

#include <iostream>
#include "ProjectFunctions.hpp"
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
    HighsSparseMatrix a;
    vector<vector<double>> A; // We will include the dense version of a matrix for now soon to go
    double s;

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
    string model_file = "/Users/akbarlatif/Desktop/scriptz/ExampleLPs/100Lp/" + model + ".mps";

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

    printf("%i non zeros, num row = %i and num col = %i \n", num_nonZeros, num_rows, num_cols);
  
    A = sparseColumn_to_full(A_start, A_index, A_value); // Again eventually the method will only take sparse
    matPrint(A);
    vector<double> x_k(num_cols, 0);
    vector<double> y_k(num_rows, 0);

    vector<double> x_k1(x_k.size(), 0);
    vector<double> y_k1(y_k.size(), 0);

    double a_n; //The soon to be ||A|| value
    a_n = sparseMatrixNorm(A_value, A_start, A_index, num_rows, num_cols);


    for(int i : A_start)
        printf("%i \t" , i);
    cout << endl;
    for(int i : A_index)
        printf("%i \t" , i);
    cout << endl;
    // = matrixNorm(A);  // ≤ 1 || in (0.31-1)
    printf("The size of argc is %i \n", argc);

    if (argc < 3)
    {
        s = 1 / a_n;
    }
    else if (stod(argv[2]) == 0)
    {
        s = 1 / a_n;
    }
    else
    {
        s = stod(argv[2]);
    }
    printf("1/||A||_2 is %g and s is %g \n", 1 / a_n, s);

    /***************************************************
        Test the PDHG Update for one update at a time (a sense check that sparse wise works)
    *****************************************************/
    cout << "Dense" << endl;
    PDHGupdate(x_k, y_k, x_k1, y_k1, s, A, b, c);
    cout << "x_k's : \n";
    vectorPrint(x_k1);
    cout << "y_k's : \n";
    vectorPrint(y_k1);
    cout << y_k1.size() << endl;

    // vector<double> dense_x_k = x_k1;
    // vector<double> dense_y_k = y_k1;
    // x_k.assign(num_cols, 0);
    // y_k.assign(num_rows, 0);
    // x_k1.assign(x_k.size(), 0);
    // y_k1.assign(y_k.size(), 0);

   vector<double> sx_k(num_cols, 0);
    vector<double> sy_k(num_rows, 0);

    vector<double> sx_k1(x_k.size(), 0);
    vector<double> sy_k1(y_k.size(), 0);
   
   
    cout << "Sparse" << endl;
    PDHGupdate_sparse(sx_k, sy_k, sx_k1, sy_k1, s, A_value, A_start, A_index, b, c);
    cout << "sparse x_k's : \n";
    vectorPrint(sx_k1);
    cout << "sparse y_k's : \n";
    vectorPrint(sy_k1);
    cout << y_k1.size() << endl;
    for(int i =0; i < y_k1.size(); i++){
        if(sy_k1[i] != y_k1[i]){
            //printf("dOh! %g does not equal %g \n", dense_y_k[i],  y_k1[i]);

            printf("dOh! %g does not equal %g and the difference is %g \n", sy_k1[i],  y_k1[i], abs(sy_k1[i] - y_k1[i]));
        }
    }
    assert(sx_k1 == x_k1);
    assert(sy_k1 == y_k1);

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



    // for (HighsInt iCol = 0; iCol < num_col; iCol++)
    //     printf("Col %1d has optimal primal value %g; dual value %g\n", int(iCol), solution.col_value[iCol], solution.col_dual[iCol]);
    // for (HighsInt iRow = 0; iRow < num_row; iRow++)
    //     printf("Row %1d has optimal primal value %g; dual value %g\n", int(iRow), solution.row_value[iRow], solution.row_dual[iRow]);

    /*****************************************
    Here are examples of the chippendale and blending models - code to be deleted soon.
    ******************************************/
    //    x_k.assign(4,0);
    //    y_k.assign(2,0);
    //    A = {{1,2,1,0}, {1,4,0,1}};
    //
    //
    //    double A_norm = matrixNorm(A);
    //    double s_standard = 1/A_norm;
    //    double s = s_standard; // < 0.2434
    //    printf("||A||_2 is %g and s is %g \n", A_norm, s_standard);
    //    vector<double> c{-10,-25,0,0};
    //    vector<double> b{80,120};

    // Blending problems
    //    x_k.assign(4,0);
    //    y_k.assign(2,0);
    //    A = {{0.3, 0.5, 1,0}, {0.7,0.5,0,1}};
    //
    //    double A_norm = matrixNorm(A);
    //    double s_standard = 1/A_norm;
    //    double s = 0.75; // < 0.82
    //    printf("||A||_2 is %g and s is %g \n", A_norm, s_standard);
    //    vector<double> c{-8,-10,0,0};
    //    vector<double> b{120,210};

    /********************************************************
    Run the PDLP model
    for now the default is solving it dense, but soon it will be solved sparse
    *********************************************************/
   
    vector<double> snx_k(num_cols, 0);
    vector<double> sny_k(num_rows, 0);

    vector<double> snx_k1(snx_k.size(), 0);
    vector<double> sny_k1(sny_k.size(), 0);
    
    
    bool up = 0;
    int iter = 0;

    if (argc < 4 || argv[3] == nullptr || strcmp(argv[3], "dense") == 0 || strcmp(argv[3], "Dense") == 0)
    {
        cout << "Running PDLP with dense matricies: \n";
        while (!up)
        {
            PDHGupdate(x_k, y_k, x_k1, y_k1, s, A, b, c);
            up = stop_update(x_k, x_k1, y_k, y_k1);
            restart_solve(x_k, x_k1, y_k, y_k1);
            iter++;
        }
    }
    else if (strcmp(argv[3], "sparse") == 0 || strcmp(argv[3], "Sparse") == 0)
    {
        // cout << "Running PDLP with Sparse matricies: \n";
        // while (!up)
        // {
        //     PDHGupdate_sparse(x_k, y_k, x_k1, y_k1, s, A_value, A_start, A_index, b, c);
        //     up = stop_update(x_k, x_k1, y_k, y_k1);
        //     restart_solve(x_k, x_k1, y_k, y_k1);
        //     iter++;
        // }
        cout << "Running PDLP with Sparse matricies: \n";
        while (!up)
        {
            PDHGupdate_sparse(snx_k, sny_k, snx_k1, sny_k1, s, A_value, A_start, A_index, b, c);
            //vectorPrint(snx_k1);
            up = stop_update(snx_k, snx_k1, sny_k, sny_k1);
            restart_solve(snx_k, snx_k1, sny_k, sny_k1);
            iter++;
        }
        vectorPrint(snx_k);
        printf("The code performed %d iterations \n", iter);
        printObjectiveValue(snx_k, c);
        for(int i = 0; i < c.size(); i++){
        if (c[i] != 0){
            printf("We have for column %i our HiGHS solution gives %g where PDHG gives us %g at cost %g \n", i, solution.col_value[i], snx_k[i],c[i]);
        }
    }
    }
    else
    {
        cout << "Try again didn't specify your model properly \n";
    }

    // printf("The code performed %d iterations \n", iter);
    // printObjectiveValue(x_k, c);
    // //printAllResults(x_k, y_k, c);
    // //cout << "Vector costs are:" << endl;
    // //vectorPrint(c);
    // for(int i = 0; i < c.size(); i++){
    //     if (c[i] != 0){
    //         printf("We have for column %i our HiGHS solution gives %g where PDHG gives us %g at cost %g \n", i, solution.col_value[i], x_k[i],c[i]);
    //     }
    // }
}
