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
#include <chrono>

using namespace std;

int main(int argc, const char *argv[]){
    
    //declare the chrono uses
    using chrono::high_resolution_clock;
    using chrono::duration_cast;
    using chrono::duration;
    using chrono::milliseconds;

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
    if (argc < 2){
        model = "avgas";
    }
    else{
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

   //Use the HiGHS getStandardFormLp() to put the given model in standard form
    h.setOptionValue("output_flag", false);
    h.getStandardFormLp(num_cols, num_rows, num_nonZeros, offsets, c.data(),
                        b.data(), A_start.data(), A_index.data(),
                        A_value.data());
    
   /*******************************************
      Run my PDLP version 
    *******************************************/
    //Print relevant information about the model
    // printf("%s has %i non zeros, num row = %i and num col = %i \n", model.c_str(), num_nonZeros, num_rows, num_cols);
    printf("PDLP solver test on %s \n", model.c_str()); //Declaration for running the solver
    
    //initialize variables 
    bool chamPock = false; //The bool for whether CP scaling will be applied
    string filePathName = "Junk 10.03";  //File name for output 
    auto t1 = high_resolution_clock::now(); // begin the runtime

    //Initialize the model as a PDLP instance
    PDLP model1;

    //Assign the values
    model1.assignLpValues(num_rows, num_cols, num_nonZeros, c, b, A_value, A_index, A_start, model, filePathName);
    model1.model = model;

    //Create relevant for the argc parsing
    debugFlag = 0;
    const char *debug = "debug";
    const char *rescale = "rescale";
    const char *input4 = "";  
    const char *input3 = argv[3];

    // Sets the iteration cap for the solver
    if(argc > 5 && stod(argv[5]) != 0) {
      max_iterations = stod(argv[5]);
      model1.iter_cap = max_iterations;   
    }

    //Determines wheter the debug information will be displayed
    if(strcmp(debug, input3) == 0) debugFlag = 1;
    model1.debugFlag = debugFlag; 
    
    //Determines wheter Chambolle-Pock scaling will be applied
    if(argc > 6 && stod(argv[6]) != 0) {
      chamPock = stoi(argv[6]);
      model1.chamPockStatus = chamPock;   
    }

    //Allows the user to put in a certain amount of ruiz rescale iterations
    if(argc > 7 && stod(argv[7]) != 0) {
      double rr_iter_cap = stod(argv[7]);
      model1.rrescale_iter_cap = rr_iter_cap;   
    }

    //Determines whether the primal and dual step sizes are equal
    if(argc > 8 && stoi(argv[8]) != 0) {
      model1.flatten_step_size = true;   
    }

    //Determines wheter the alternative model of scaling will be applied
    if(argc > 9 && stoi(argv[9]) != 0) {
      model1.alternate_Scaling = true;   

    }

    //Determines whether the solver will be run with or without rescaling
    if(argc > 4){
        input4 = argv[4];
        if(strcmp(rescale, input4) == 0){
        model1.statusRescale = 1; 
        model1.run_Rescale();
        } 
    } 

    //Allows the user to input a certain stepsize
    if(argc > 2 && stod(argv[2]) != 0) {
        s = stod(argv[2]);
        model1.step_size = s;   
    }
    
    //initializes norms and stepsizes
    model1.initialiseModel();
   
    //Runs the model until an optimal value is found
    model1.runFeasiblePDHG(1, debugFlag);

    //Prints the optimal objective value
    model1.printObjectiveValue();

    //Stops the runtime once the optimal value is displayed
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count()/1000 << "s\n";
    double runTime = ms_double.count();
    
    //Writes the optimal solution value as well as number of iterations and runtime to a csv file
    model1.writeFile(runTime);

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
    // printf("Num of HiGHS rows %i, Num of HiGHS cols %i \n", num_row, num_col);
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
    h.run();
    HighsSolution solution = h.getSolution();
    cout << endl;
   
}
   

  
