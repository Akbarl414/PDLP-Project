//
//  ProjectFunctions.cpp
//  MathsProjectCode`
//
//  Created by Akbar Latif on 10/18/24.
//
#include <iostream>
#include "ProjectFunctions.hpp"
#include "PDLPClass.hpp"
#include <vector>
#include <math.h>
#include <cassert>
#include <cmath>
#include <fstream>
// #include Highs.h

using namespace std;

// A function that prints the input matrix of any size when called
void matPrint(vector<vector<double>> &mat)
{
    for (int i = 0; i < mat.size(); i++)
    {
        for (auto j = 0; j < mat[0].size(); j++)
        {
            cout << mat[i][j] << "\t";
        }
        cout << endl << endl;
    }
}

// A function that prints a the input 2x1 array when called
void vectorPrint(vector<double> &arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        cout << arr[i] << "\t";
    }
    cout << endl;
}
//A dunction that prints a vector of type ints
void vectorPrint(vector<int> &arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        cout << arr[i] << "\t";
    }
    cout << endl;
}

//This function initialises all of the relevant data for the LP instance
void PDLP::assignLpValues(int &num_row, int &num_col, int &num_nonZero, vector<double> &cost,
    vector<double> &bound, vector<double> &lp_matrix_values, vector<int> &lp_matrix_index, 
    vector<int> &lp_matrix_start, string modelName, string filePathName){
    //Initialise the information for Matrix A
    num_rows = num_row; 
    num_cols = num_col;
    num_nonZeros = num_nonZero;
    matrix_values = lp_matrix_values;
    matrix_index = lp_matrix_index;
    matrix_start = lp_matrix_start;

    //The costs and bounds of the LP
    costs = cost;
    bounds = bound;

    //Assign the relevant information for the primal and dual updates 
    x_k.assign(num_cols, 0);
    x_k1.assign(num_cols, 0);
    y_k.assign(num_rows, 0);
    y_k1.assign(num_rows, 0);

    //Size vairous instances for feasibility and rescale checks
    scaled_matrix_values.resize(num_nonZeros);
    reducedCosts.assign(num_cols, 0);
    
    //Other information for the Update functions
    feasibility_tolerance = 10E-4;  // Make it 10E-4 or 10E-8 (which is the square of 10E-4) 
    iter_cap = 2.5E6;
    
    //Information for the output functions
    model = modelName; 
    filePath = filePathName;
}

//A function that tells the model which matrix rescaling to perform based on user input.
void PDLP::run_Rescale(){
    //If the user has selected alternate scaling run the alternative versions of the rescaling methods
    if(alternate_Scaling){
        ruiz_Rescale_alternate();
        if(chamPockStatus) chamPock_Rescale_alternate();
    }
    else{ //Run the normal versions
        ruiz_Rescale();
        if(chamPockStatus) chamPock_Rescale();
    }
    //Scale the rest of the LP after resclaing
    scaleLP();
}

//A function that initializes the norms and other parts of the solver
void PDLP::initialiseModel(){
    //Calculate the stepsize using the power method if there is no user inputted step size
    if(step_size == 0){
        matrix_norm = matrixNorm();
        step_size = 1/matrix_norm; //have the step size be 1/||A||_2
    }
    initialiseNorms(); // Calculate the norms of the model

    //Intialize the primal and dual step sizes
    primal_feasibility_tolerance = feasibility_tolerance * (1 + bounds_2_norm);
    dual_feasibility_tolerance  = feasibility_tolerance * (1 + costs_2_norm);
    primal_step_size = step_size/step_balance;
    dual_step_size = step_size * step_balance;
    //If the step_balance is zero or the user has specified a flattened step size have primal and dual step sizes equal the global step size
    if(step_balance == 0 || flatten_step_size){ 
        dual_step_size = step_size; 
        primal_step_size = step_size;
    }
    //Output the model's tolerances and steps sizes
    printf("The dual_feas_tolerance = %g, and prim_feas_tol = %g, dual_s_s = %g and prim_s_s = %g. \n", dual_feasibility_tolerance, primal_feasibility_tolerance,dual_step_size, primal_step_size);
}

//A function to initialize the cost and bound norms as well as the step balance (primal weight)
void PDLP::initialiseNorms(){
    //create variables to be the sum_i x_i^2 for the two norm of the vectors
    double bounds_squared_err = 0; 
    double costs_squared_err = 0; 
    
    //Calculate the 2-norms of bounds and costs
    for(int iCol = 0; iCol < num_cols; iCol++){
        costs_squared_err += pow(costs[iCol], 2);
    }
    for(int iRow = 0; iRow < num_rows; iRow++){
        bounds_squared_err += pow(bounds[iRow],2);
    }
    bounds_2_norm = sqrt(bounds_squared_err);
    costs_2_norm = sqrt(costs_squared_err); 

    //initialize the step balance(primal weight)
    step_balance = costs_2_norm/bounds_2_norm;
    // printf("Bounds norm is %g \n", bounds_2_norm);
    assert(bounds_2_norm > 10E-16);
}

//A function the uses the power method to calculate the two-norm of an mxn matrix A
double PDLP::matrixNorm(){
    //initialize the relevant varibles
    vector<double> xk, w, z;
    xk.assign(num_cols, 1);
    w.resize(num_cols);
    int iter = 0;
    double w_norm = 0;
    double dl_x_norm = 0;
    //Perform the power method
    for (;;){
        // Form z = Ax_k
        z.assign(num_rows, 0);
        for (int iCol = 0; iCol < num_cols; iCol++){
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
                z[matrix_index[iEl]] += matrix_values[iEl] * xk[iCol];
            }
        }
        
        //Form w = A^tz
        for (int iCol = 0; iCol < num_cols; iCol++){
            w[iCol] = 0;
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){  
                  w[iCol] += matrix_values[iEl] * z[matrix_index[iEl]];
            }
        }
        
        //Calculate the infinity norm of w
        w_norm = 0;
        for (int iCol = 0; iCol < num_cols; iCol++){
            w_norm = max(abs(w[iCol]), w_norm);
        }
        //if w norm is non-positive then return 1 as the value
        if(w_norm <= 0) return 1;

        //Check to see if the values are converging or not
        dl_x_norm = 0;
        for (int iCol = 0; iCol < num_cols; iCol++){
            w[iCol] /= w_norm;
            dl_x_norm = max(abs(w[iCol] - xk[iCol]), dl_x_norm);
        }
        xk = w;
        //if they have converged end the method
        if (iter > 1000 || dl_x_norm < 1e-10)
            break;
        iter++;
    }
    //printf("Iteration %4d: w_norm = %g; dl_x_norm = %g\n", int(iter), w_norm, dl_x_norm);
    if(debugFlag) printf("||A||_2 = %g\n", sqrt(w_norm));
    return sqrt(w_norm);
}

//Create a function to implement the PDHG updates 
void PDLP::PDHGUpdate(){
    //initialize AtY
    vector<double> AtYk(num_cols, 0);
    
    //calculate AtY
    for (int iCol = 0; iCol < num_cols; iCol++){
        double value = 0;
        for (int columnK = matrix_start[iCol]; columnK < matrix_start[iCol + 1]; columnK++){
            int iRow = matrix_index[columnK];
            value += matrix_values[columnK] * y_k[iRow];
        }
        AtYk[iCol] = value;
    }

    //perform the primal update
    for (int iCol = 0; iCol < num_cols; iCol++){
        x_k1[iCol] = ((x_k[iCol] + primal_step_size * AtYk[iCol] - primal_step_size * costs[iCol]));
        if (x_k1[iCol] < 0){
            x_k1[iCol] = 0;
        }
    }

    // Similarly to how we did for x_k+1 we need to do the matrix-vector multiplication
    // first to solve s*A*(x_k+1 - x_k)
    vector<double> s_Ax(num_rows, 0);
    
    for (int iCol = 0; iCol < num_cols; iCol++){
        for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++){
            s_Ax[matrix_index[column]] += dual_step_size * matrix_values[column] * (2 * x_k1[iCol] - x_k[iCol]);
        }
    }

    //Perform the dual update
    for (int iRow = 0; iRow < num_rows; iRow++){
        y_k1[iRow] = y_k[iRow] - s_Ax[iRow] + dual_step_size * bounds[iRow];
    }
}

//A Function that return the adaptive step size of an iteration (In progress)
double PDLP::adaptive_step(){
    double x_norm_sq = 0;
    double y_norm_sq = 0;
    double x_norm = 0; 
    double y_norm = 0;

    vector<double> Ax(num_rows, 0);
    // vector<double> ytAx(num_rows, 0);
    double ytAx = 0 ;
    for (int iCol = 0; iCol < num_cols; iCol++)
    {
        double x_diff = x_k1[iCol] - x_k[iCol];
        for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++)
        {
            Ax[matrix_index[column]] += matrix_values[column] * (x_diff);
        }
        x_norm_sq += pow((x_diff), 2); 
    }

    for(int iRow = 0; iRow < num_rows; iRow ++){
        double y_diff = y_k1[iRow] - y_k[iRow];
        // printf("y_diff is %g and Ax[iRow] is %g \t", y_diff, Ax[iRow]);
        y_norm_sq += pow((y_diff), 2);
        ytAx += y_diff * Ax[iRow];
    }
    // if(ytAx < 10E-12) return step_size;
    // printf("ytAx is %g \n", ytAx);
    double numerator = step_balance *x_norm_sq + y_norm_sq/ step_balance;
    double adaptive_step_value = numerator / (2*ytAx);
    // printf("adapative step size is %g \n", adaptive_step_value);
    return adaptive_step_value;
    
}

//A helper function used to subtract two vectors 
vector<double> PDLP::vectorSubtraction(vector<double> &vect1, vector<double> &vect2){
    assert(vect1.size() == vect2.size());
    vector<double> result(vect1.size(), 0);
    for (int index = 0; index < vect1.size(); index++){
        result[index] = vect1[index] - vect2[index];
    }
    return result;
}

//A helper function to take the 2-norm of a function
double PDLP::vectorEuclidianNorm(vector<double> &vect){
    double norm_squared = 0;
    for (int index = 0; index < vect.size(); index++){
        norm_squared += pow(vect[index], 2);
    }
    return sqrt(norm_squared);
}  

//The old update criteria for a model that terminates when updates do not change vastly between iterations
bool PDLP::updateCriteria(){
    vector<double> x_norm_inside = vector_Subtraction(x_k1, x_k);
    vector<double> y_norm_inside = vector_Subtraction(y_k1, y_k);

    double update_criteria_x = vectorNorm(x_norm_inside);
    double update_criteria_y = vectorNorm(y_norm_inside);

    if (update_criteria_x > 0.00001 && update_criteria_y > 0.00001){
        return 0;
    }
    else{
        return 1;
    }
}

//A function that runs the old version of PDHG using the updateCriteria() functions checking for iterate difference rather than feasibility checks
void PDLP::runPDHG(bool outputFlag)
{
    if (outputFlag == 1) printf("Running Resatarted PDHG with a step size of %g \n", step_size);
    up = 0;
    iterations = 0;
    while(!up)
    {
        PDHGUpdate();
        up = updateCriteria();
        restartSolve();
        iterations ++;
    }
    getObjectiveValue();
    if(outputFlag ==1) printf("Restarted PDHG ran with %i interations \n", iterations);
}

//A function to run PDHG with feasibility checks rather than iteration differences
void PDLP::runFeasiblePDHG(bool outputFlag, bool debugFlagValue){
    //initializaions running the model
    debugFlag = debugFlagValue; 
    up = 0;
    iterations = 0;

    //Iterate until the iterations are feasible
    while(!up){
        //update the primal and dual
        PDHGUpdate();
        //reset the primal and dual values
        restartSolve();
    
        // addResults(); //to add the each iterations values to the csv output
        //Check the feasibility 
        up = isFeasible();
        iterations ++;
    }
    
    //For debugging
    if(outputFlag ==1) {
        if(statusRescale){
            printf("For the scaled model we have: \n");
            isComplementarity();
            printObjectiveValue();
        }
    }

    //If the model was scaled scale the LP back to the original values and calculate the feasibility checks for the unscaled values 
    if(statusRescale){
        unScaleLP();
        calculateReducedCosts();
        isDualFeasible();
        isPrimalFeasible();
        isDualityGap();
        isComplementarity();
    } 
    //Calculate the objective value
    getObjectiveValue();
    //Print statement before the final results
    printf("In the end we have: \n");
}

//A function to add primal and dual iterates to the csv output
void PDLP::addResults(){
    for(int iCol = 0; iCol < num_cols; iCol ++){
        x_k_result.push_back(x_k[iCol]);
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        y_k_result.push_back(y_k[iRow]);
    }
}

//A function to write the output to csv using fstream
void PDLP::writeFile(double &runTime){
    fstream MyFile;
    MyFile.open(filePath + ".csv", ios::app);

    // MyFile << "Model, Obj, Iterations, Duality Gap, Complementarity, Dual feasibility, Primal relative feasibility, num_rows, num_cols, num_nonZeros, Dual_feas_tolerance, Primal_feas_tolerance, Run Time (s) \n";
    MyFile << model.c_str() << ","<< objectiveValue<< ","<<iterations <<  "," << dualityGap<< "," << adjusted_complementarity << ","<<dual_inf_norm<< ","<<primal_relative_error<< "," << num_rows<< ","<< num_cols<<"," << num_nonZeros <<"," << dual_feasibility_tolerance <<"," << primal_feasibility_tolerance<< "," << runTime/1000<<"," << endl;
    MyFile.close();

    //Code for the 'toy' example
    // fstream ResultFile;
    // ResultFile.open("Lu24_xy.csv", ios::app);
    // ResultFile << "X_values" << "," << "Y_values" << "," << "Iteration" << "," << endl;
    // for(auto index = 0; index < x_k_result.size(); index ++){
    //     ResultFile << x_k_result[index] << "," << y_k_result[index] << "," << index << "," <<endl;
    // }
    // ResultFile.close();
}

//A function that calculates the objective value 
void PDLP::getObjectiveValue(){
    double resultValue; 
    for (int iCols = 0; iCols < num_cols; iCols++){
            resultValue += costs[iCols] * x_k[iCols];
        }
    objectiveValue = resultValue;
}

//A function that prints the object value and relevant results of the model
void PDLP::printObjectiveValue(){
    getObjectiveValue();
    if(!up) cout << "Make sure you have Run the model \n"; 
    else{
        printf("Model, Obj, Iterations, Duality Gap, Complementarity, Dual feasibility, and Primal relative feasibility is: %s, %g, %i, %g, %g, %g, %g \n", model.c_str(), objectiveValue, iterations, dualityGap, adjusted_complementarity, dual_inf_norm, primal_relative_error);
    }   
}
//A function that prints the objective value as well as the optimal primal and dual values 
void PDLP::printFullResults(){
    printf("Our optimal objective value is: %g \n", objectiveValue);
    cout << "Optimal Primal values \n";
    for (double values : x_k){
        printf("%g \t", round(values));
    }
    cout << endl;
    cout << "Optimal Dual values" << endl;
    for (double values : y_k){
        printf("%g \t", round(values));
    }
    cout << endl;
}

//A function that sets xk = xk+1 and yk = yk+1 for each iteration
void PDLP::restartSolve(){
    for (int i = 0; i < num_cols; i++){
        x_k[i] = x_k1[i];
        x_k1[i] = 0;
    }
    for (int j = 0; j < num_rows; j++){
        y_k[j] = y_k1[j];
        y_k1[j] = 0;
    }
    // reset the step sizes for 
    primal_step_size = step_size/step_balance;
    dual_step_size = step_size * step_balance;
}

//A function that calculates the reduced costs vector lambda
void PDLP::calculateReducedCosts(){
    //Calculate vector AtY
    vector<double> AtYt(num_cols, 0);
    for (int iCol = 0; iCol < num_cols; iCol++){
        double value = 0;
        for (int columnK = matrix_start[iCol]; columnK < matrix_start[iCol + 1]; columnK++){
            int iRow = matrix_index[columnK];
            value += matrix_values[columnK] * y_k[iRow];
        }
        AtYt[iCol] = value;
    }

    //Calculate the reduced costs as proj_R^+(c - AtY)
    for(int iCol = 0; iCol < num_cols; iCol++){
        double reducedCosti = costs[iCol] - AtYt[iCol];
        if(-reducedCosti < 0){
            reducedCosts[iCol] = reducedCosti; 
        }
        else reducedCosts[iCol] = 0;
    }
}

//A function that checks for primal feasibility and returns whether ||Ax - b||_2 < relative primal tolerance 
bool PDLP::isPrimalFeasible(){
    //initialize local variables
    double absolute_error;
    double bound_square_error = 0;
    primal_relative_error = 0;
    vector<double> Ax(num_rows, 0);

    //construct ||Ax-b||_2
    for (int iCol = 0; iCol < num_cols; iCol++){
        for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++){
            Ax[matrix_index[column]] += matrix_values[column] * x_k[iCol];
        }
    }
    // Check primal feasibility 
    primal_2_norm = 0;
    for (int iRow = 0; iRow < num_rows; iRow++){
        absolute_error = 0;
        absolute_error = abs(Ax[iRow] - bounds[iRow]);
        primal_2_norm += pow(absolute_error, 2);
    } 

    primal_relative_error = sqrt(primal_2_norm);
    if (primal_relative_error > primal_feasibility_tolerance) return false;  // Not feasible
    return true;  // Primal feasible
}

/******************* 
This version of the Dual feasibilty takes the inf_norm 
********************/
// bool PDLP::isDualFeasible() {
//     bool dual_feasibility = 0; 
//     dual_inf_norm = 100; 
//     for(int iCol = 0; iCol < num_cols; iCol++ ){       
//         dual_inf_norm = min(reducedCosts[iCol], dual_inf_norm); 
//         // if((reducedCosts[iCol]/costs_2_norm) < -feasibility_tolerance) return false;
//     }
//     if(dual_inf_norm < -feasibility_tolerance*(1+ costs_2_norm)) return false;
//     return true;
// }
/******************* 
This version of the Dual feasibilty takes the two_norm 

********************/

// A function that returns a Bool value if |c - A^Ty - \lamda| < relative dual tolerance 
bool PDLP::isDualFeasible(){
    //initialize variables 
    dual_inf_norm = 0;
    double dual_feasibility = 0; 
    vector<double> AtYt(num_cols, 0);
    
    //compute AtY
    for (int iCol = 0; iCol < num_cols; iCol++){
        double value = 0;
        for (int columnK = matrix_start[iCol]; columnK < matrix_start[iCol + 1]; columnK++){
            int iRow = matrix_index[columnK];
            value += matrix_values[columnK] * y_k[iRow];
        }
        AtYt[iCol] = value;
    }

    //calculate ||c - A^Ty - \lamda||_2
    for(int iCol = 0; iCol < num_cols; iCol++ ){       
        dual_feasibility += abs(costs[iCol] - AtYt[iCol] - reducedCosts[iCol]);        
    }
    dual_inf_norm = sqrt(max(dual_feasibility, 10E-20));
    //check feasibility 
    if(dual_inf_norm > dual_feasibility_tolerance) return false;
    return true;
}

//A function that check whether a given x,y pair is optimal using relative complementarity
bool PDLP::isComplementarity(){
    //initialize the varibables
    complementarity = 0;
    adjusted_complementarity = 0;
    //compute sum_i x_i*lambda_i  
    for(int iCol = 0; iCol < num_cols; iCol++ ){
        complementarity += x_k[iCol]*abs(reducedCosts[iCol]);
        adjusted_complementarity += x_k[iCol]*abs(reducedCosts[iCol]/costs_2_norm);
    }
    //check complementarity
    if (adjusted_complementarity < feasibility_tolerance) return true;  
    return false; 
}

//A function that checks wheter the duality gap < \epsilon
bool PDLP::isDualityGap(){
    //initialize varibales 
    double btY = 0;
    double ctX = 0;

    //construct the object values for the dual and primal
    for(int iCol = 0; iCol < num_cols; iCol++){
        ctX += x_k[iCol] * costs[iCol];
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        btY += bounds[iRow] * y_k[iRow];
    }
    //compute duality gap
    dualityGap = abs(btY - ctX);
    //check for relative tolerance
    if(dualityGap < feasibility_tolerance*(1 + abs(ctX) + abs(btY))) return true;
    return false;
}

//A function that returns whether a given iterate satisfies primal and dual feasibility as well as complementarity
bool PDLP::isFeasible(){
    calculateReducedCosts();
    if(isPrimalFeasible() && isDualityGap() && isDualFeasible()) return true;
    // if(isPrimalFeasible() && isComplementarity() && isDualFeasible()) return true;
    if(debugFlag && iterations%20000 == 0) //Include the norms and completmentarity 
    {
        getObjectiveValue();
        isPrimalFeasible(); isDualityGap(); isDualFeasible(); isComplementarity();
        // printf("Obj, Iterations, adjusted_Complmentarity, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g \n", objectiveValue,iterations, adjusted_complementarity, dual_inf_norm, primal_relative_error);
        printf("Obj, Iterations, Duality Gap, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g \n", objectiveValue,iterations, dualityGap, dual_inf_norm, primal_relative_error);
    }
    if(iterations > iter_cap) {
        // printf("Obj, Iterations, Complementarity, adjusted_Complmentarity, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g, %g \n", objectiveValue,iterations, complementarity, adjusted_complementarity, dual_inf_norm, primal_relative_error);
        printf("Obj, Iterations, Duality Gap, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g \n", objectiveValue,iterations, dualityGap, dual_inf_norm, primal_relative_error);
        printDebug();
        return true;
    }    
    return false;
}

//A function that was used for debugging primal feasibility
void PDLP::printDebug()
{
    if(debugFlag){
        double absolute_error;
        vector<double> Ax(num_cols, 0);
        for (int iCol = 0; iCol < num_cols; iCol++){
            for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++){
                Ax[matrix_index[column]] += matrix_values[column] * x_k[iCol];
            }
        }

        double two_norm_squared = 0;
        for (int iCol = 0; iCol < num_cols; iCol++){
        absolute_error = abs(Ax[iCol] - bounds[iCol]);
        two_norm_squared += pow(absolute_error, 2);
        primal_relative_error = max(primal_relative_error, absolute_error);
        }
    }    
}

//A function that uses Daniel Ruiz's method of matrix scaling to scale our LP
void PDLP::ruiz_Rescale(){ 
    //Initialize all relevant valiables
    const double scale_epsilon = 1e-12;
    diag_c.assign(num_cols,0);
    diag_r.assign(num_rows,0);
    scaled_matrix_values = matrix_values;
    d_r.assign(num_rows, 1);
    d_c.assign(num_cols, 1);
    int rr_iterations = 1;

    //Perform the rescaling
    for(int rescale_iterations = 0;rescale_iterations < rrescale_iter_cap; rescale_iterations ++){
        //Construct the value vectors for the diagonal matricies
        for (int iCol = 0; iCol < num_cols; iCol++){
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
                int iRow = matrix_index[iEl];
                diag_c[iCol] = max(fabs(scaled_matrix_values[iEl]), diag_c[iCol]);
                diag_r[iRow] = max(fabs(scaled_matrix_values[iEl]), diag_r[iRow]);
            }
            //verify that no columns have 0 values
            if(diag_c[iCol] == 0){
                diag_c[iCol] = 1; 
            } 
            diag_c[iCol] = sqrt(diag_c[iCol]);
            diag_c[iCol] = 1/diag_c[iCol];
        }
        
        for(int iRow = 0 ; iRow < num_rows; iRow ++ ){
            if(diag_r[iRow]  == 0){
                diag_r[iRow] = 1;
                // printf("We have a zero at row %i \n", iRow);
            } 
            // diag_r[iRow] = sqrt(max(diag_r[iRow], scale_epsilon)); //make diag_r = sqrt(D_R)
           diag_r[iRow] = sqrt(diag_r[iRow]);
            // if(diag_r[iRow] < scale_epsilon) printf("uh oh diag_r is %g \n", diag_r[iRow]);
            diag_r[iRow] = 1/diag_r[iRow];
        }

        //Multiply the values of A~ = A_k * D_C^-1
        scale_Column(diag_c);

        //Now multiplying the  A_hat = D_R^-1 * A~     
        scale_Row(diag_r);

        //For the output create min and max values to see convergence of the scaling factors
        double max_dr = 0; 
        double max_dc = 0;
        double min_dr = 1;
        double min_dc = 1;
        
        //Set the d_r and d_c values for the next iteration
        for(int iCol = 0; iCol < num_cols; iCol ++){
            double one_over = diag_c[iCol];
            d_c[iCol] = d_c[iCol]*(one_over); 
            max_dc = max(one_over, max_dc);
            min_dc = min(one_over, min_dc);
            diag_c[iCol] = 0;
        }
        for(int iRow = 0; iRow < num_rows; iRow ++){
            double value = diag_r[iRow];
            d_r[iRow] = d_r[iRow]*(value);
            // if(value > max_dr) printf("on index %i, the row scaling number %g is bigger than %g \n", iRow, value, max_dr);
            max_dr = max(value, max_dr);
            min_dr = min(value, min_dr);
            diag_r[iRow] = 0;
        }
        if(debugFlag) printf("In iteration %i D_rk takes values between (%g,%g) and D_ck takes (%g,%g) \n", rr_iterations, min_dr, max_dr, min_dc, max_dc);
        rr_iterations ++;
    }
    if(debugFlag) printf("The Ruiz Rescaling went through %i iterations \n", rr_iterations);
   
    //A sense check :
    vector<double> sense_check(num_nonZeros);
    for(int iCol = 0; iCol < num_cols; iCol ++){
        for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl ++){
            sense_check[iEl] =  matrix_values[iEl]*d_c[iCol];
        }            
    }
    for(int iCol = 0 ; iCol < num_cols; iCol ++ ){
            for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol +1]; iEl++){
                double value = sense_check[iEl];
                double multiplier = d_r[matrix_index[iEl]];
                sense_check[iEl] = value*(multiplier);    
        }   
    }
    for(int index = 0; index < num_nonZeros; index++ ){
        if(abs(scaled_matrix_values[index]-sense_check[index]) > 1E-2) printf("Index %i with the difference %g \t ", index, abs(scaled_matrix_values[index]-sense_check[index]));
        // if(abs(scaled_matrix_values[index]) > 1.00000000001) printf("Index %i at value %g \t ", index, abs(scaled_matrix_values[index]));
    }
}

//Helper function to scale the columns of the matrix
void PDLP::scale_Column(vector<double> &scalingVector){
    for(int iCol = 0; iCol < num_cols; iCol ++){
        for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl ++){
            scaled_matrix_values[iEl] = scaled_matrix_values[iEl]*(scalingVector[iCol]); 
        }            
    }

}

//Helper function to scale the rows of the matrix
void PDLP::scale_Row(vector<double> &scalingVector){
    for(int iCol = 0 ; iCol < num_cols; iCol ++ ){
        for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol +1]; iEl++){
            double value = scaled_matrix_values[iEl];
            double multiplier = scalingVector[matrix_index[iEl]];
            scaled_matrix_values[iEl] = value*(multiplier);    
        }   
    }
}

//A function to perform chambolle-pock diagonal preconditioning
void PDLP::chamPock_Rescale(){
    double max_row_value = 0;
    printf("Beginning the Chambolle-Pock rescaling \n ");
    //Need to compute the 1-norm for the columns and the rows, then apply them to the matrix A
    // 1- norm of the rows :
    col_norm.assign(num_cols, 0);
    row_norm.assign(num_rows, 0);
    for (int iCol = 0; iCol < num_cols; iCol++){
        for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
            //Accumulate the 1- norm of the columns

            col_norm[iCol] += fabs(scaled_matrix_values[iEl]);
            
            //Now for the rows
            int iRow = matrix_index[iEl];
            row_norm[iRow] += fabs(scaled_matrix_values[iEl]);
        }
        if(col_norm[iCol] ==0) col_norm[iCol] = 1;
        col_norm[iCol] = 1 / sqrt(col_norm[iCol]);
    }
    //compute the sqrt of the 1- norm and set the scaling factor to 1/1-norm
    for(int iRow = 0; iRow < num_rows; iRow++){
        if(row_norm[iRow] ==0) row_norm[iRow] = 1;
        double value = sqrt(row_norm[iRow]);
        row_norm[iRow] = 1/ value;

        max_row_value = max(row_norm[iRow], max_row_value);
    }

    //Multiply the values of A~ = A_k * D_C^-1
    scale_Row(row_norm);
    scale_Column(col_norm);
    printf("the max row value for CP scaling is %g \n ", max_row_value);
}

//A function that scales the LP after performing scaling on the matrix
void PDLP::scaleLP(){
    if(!chamPockStatus){  //If chambolle-pock scaling is not used have its scaling values be 1
        col_norm.assign(num_cols, 1);
        row_norm.assign(num_rows, 1);
    }
    //set original versions of the vectors to use post solving
    orignal_costs = costs;
    orignal_bounds = bounds;
    orignal_matrix_values = matrix_values;

    matrix_values = scaled_matrix_values; //set the matrix values to the scaled values
    //scale the costs and the bounds by the Ruiz and CP scaling vectors
    for(int iCol = 0; iCol < num_cols; iCol ++){
        double scale_value = col_norm[iCol]  * costs[iCol];
        scale_value = scale_value * d_c[iCol];
        costs[iCol] = scale_value ; 
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        bounds[iRow] = (d_r[iRow])* (row_norm[iRow]) * bounds[iRow];
    }
}

//A function that performs the scaling back to the original LP values
void PDLP::unScaleLP(){
    //set the primal and dual solution vectors to the correct size
    post_scaled_x_k.resize(num_cols);
    post_scaled_y_k.resize(num_rows);
    
    //Calculate the post-scaled primal and dual values
    for(int iCol = 0; iCol < num_cols; iCol ++){
        post_scaled_x_k[iCol] =  (col_norm[iCol])*(d_c[iCol])*x_k[iCol];
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        post_scaled_y_k[iRow] = (row_norm[iRow])*(d_r[iRow])*y_k[iRow];
    }

    //reset all relevant parts of the LP to perform the feasibility checks
    bounds = orignal_bounds;
    costs = orignal_costs; 
    x_k = post_scaled_x_k;
    y_k = post_scaled_y_k;
    matrix_values = orignal_matrix_values;
}

//An alternative version of Ruiz rescaling that performs sucessive row then column scaling each iteration 
void PDLP::ruiz_Rescale_alternate(){ 
    printf("Running the alternative Ruiz Rescaling \n");
    //Initialize all relevant valiables
    const double scale_epsilon = 1e-12;
    diag_c.assign(num_cols,0);
    diag_r.assign(num_rows,0);
    scaled_matrix_values = matrix_values;
    d_r.assign(num_rows, 1);
    d_c.assign(num_cols, 1);

    // printf("The maximum amount of iterations is %g \n", rrescale_iter_cap);
    int rr_iterations = 1;
    for(int rescale_iterations = 0;rescale_iterations < rrescale_iter_cap; rescale_iterations ++){
        //Construct the value arrays for the diagonal matricies
         for (int iCol = 0; iCol < num_cols; iCol++){
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
                int iRow = matrix_index[iEl];
                diag_r[iRow] = max(fabs(scaled_matrix_values[iEl]), diag_r[iRow]);
            }
        }
        for(int iRow = 0 ; iRow < num_rows; iRow ++ ){
            if(diag_r[iRow]  == 0){
                diag_r[iRow] = 1;
                // printf("We have a zero at row %i \n", iRow);
            } 
            diag_r[iRow] = sqrt(diag_r[iRow]); //make diag_r = sqrt(D_R)
            diag_r[iRow] = 1/diag_r[iRow];
        }
        //Now multiplying the  A_hat = D_R^-1 * A~     
        scale_Row(diag_r);
       
       
       
        for (int iCol = 0; iCol < num_cols; iCol++){
            // diag_c[iCol] = 0;
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
                int iRow = matrix_index[iEl];
                diag_c[iCol] = max(fabs(scaled_matrix_values[iEl]), diag_c[iCol]);
                diag_r[iRow] = max(fabs(scaled_matrix_values[iEl]), diag_r[iRow]);
            }
            if(diag_c[iCol] == 0){
                diag_c[iCol] = 1; 
                // printf("We have a zero at col %i \n", iCol);
            } 
            diag_c[iCol] = sqrt(diag_c[iCol]); //make diag_c = sqrt(D_C)
            diag_c[iCol] = 1/diag_c[iCol];
        }
        //Multiply the values of A~ = A_k * D_C^-1
        scale_Column(diag_c);

      

        double max_dr = 0; 
        double max_dc = 0;
        double min_dr = 1;
        double min_dc = 1;
        //Set the d_r values for the next iteration
        for(int iCol = 0; iCol < num_cols; iCol ++){
            double one_over = diag_c[iCol];
            d_c[iCol] = d_c[iCol]*(one_over); 
            max_dc = max(one_over, max_dc);
            min_dc = min(one_over, min_dc);
            diag_c[iCol] = 0;
        }
        for(int iRow = 0; iRow < num_rows; iRow ++){
            double value = diag_r[iRow];
            d_r[iRow] = d_r[iRow]*(value);
            max_dr = max(value, max_dr);
            min_dr = min(value, min_dr);
            diag_r[iRow] = 0;
        }
        if(debugFlag) printf("In iteration %i D_rk takes values between (%g,%g) and D_ck takes (%g,%g) \n", rr_iterations, min_dr, max_dr, min_dc, max_dc);
        rr_iterations ++;
    }
    if(debugFlag) printf("The Ruiz Rescaling went through %i iterations \n", rr_iterations);
   
    //A sense check :
    vector<double> sense_check(num_nonZeros);
    for(int iCol = 0; iCol < num_cols; iCol ++){
        for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl ++){
            sense_check[iEl] =  matrix_values[iEl]*d_c[iCol];
        }            
    }
    for(int iCol = 0 ; iCol < num_cols; iCol ++ ){
            for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol +1]; iEl++){
                double value = sense_check[iEl];
                double multiplier = d_r[matrix_index[iEl]];
                sense_check[iEl] = value*(multiplier);    
        }   
    }
}

//The alternative scaling version of the chambolle pock scaling function using successive row then column scaling
void PDLP::chamPock_Rescale_alternate(){
    printf("Beginning the Chambolle-Pock rescaling \n ");
    //Need to compute the 1-norm for the columns and the rows, then apply them to the matrix A
    // 1- norm of the rows :
    col_norm.assign(num_cols, 0);
    row_norm.assign(num_rows, 0);
    for (int iCol = 0; iCol < num_cols; iCol++){
        for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
            int iRow = matrix_index[iEl];
            col_norm[iCol] += abs(scaled_matrix_values[iEl]);
        }
        if(col_norm[iCol] ==0) col_norm[iCol] = 1;
        col_norm[iCol] = 1 / sqrt(col_norm[iCol]);
    }
    
    scale_Column(col_norm);

    
    //Now compute the row_norms based on the already column sclaed matrix.
    for (int iCol = 0; iCol < num_cols; iCol++){
        for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
            int iRow = matrix_index[iEl];
            row_norm[iRow] += abs(scaled_matrix_values[iEl]);
        }
    }
    for(int iRow = 0; iRow < num_rows; iRow++){
        if(row_norm[iRow] ==0) row_norm[iRow] = 1;
        double value = sqrt(row_norm[iRow]);
        row_norm[iRow] = 1 / value;
    }
    scale_Row(row_norm);
}



/*******************************
 * All of the following function are helper functions used for debugging not for the solver 
 * they are written for dense matrices but were useful when needing to look at total matrices rather than sparse versions
 *******************************/

// A Helper function that RETURNS the transpose of the inputted dense matrix
vector<vector<double>> transposeMatrix(vector<vector<double>> &mat){
    vector<vector<double>> trans(mat[0].size(), vector<double>(mat.size(), 0));
    for (int i = 0; i < trans.size(); i++){
        for (auto j = 0; j < trans[0].size(); j++){
            trans[i][j] = mat[j][i];
        }
    }
    return trans;
}


// A function to calculate the Euclidian norm ||n||_2 of any inputted vector
double vectorNorm(vector<double> &vector){
    double norm_squared = 0;
    for (int i = 0; i < vector.size(); i++){
        norm_squared += pow(vector[i], 2);
    }
    return sqrt(norm_squared);
}

// A function that returns the difference of two input vectors
vector<double> vector_Subtraction(vector<double> &vect1, vector<double> &vect2){
    if (vect1.size() != vect2.size()){
        return vector<double>(2, 0);
    }
    vector<double> result(vect1.size(), 0);
    for (int i = 0; i < vect1.size(); i++){
        result[i] = vect1[i] - vect2[i];
    }
    return result;
}


// A function that RETURNS the product of the inputted matricies, given their dimensions fit
vector<vector<double>> matrixMult(vector<vector<double>> &mat1, vector<vector<double>> &mat2){
    vector<vector<double>> result(mat1.size(), vector<double>(mat2[0].size(), 0));
    for (int i = 0; i < mat1.size(); i++){
        for (int j = 0; j < mat2[0].size(); j++){
            for (int k = 0; k < mat1[0].size(); k++){
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

// Boolean to check if it a square matrix
bool isSquareMatrix(vector<vector<double>> &mat){
    if (mat[0].size() != mat.size()){
        return 0;
    }
    else{
        return 1;
    }
}

// A function to take a row spare matrix and make it into a full matrix
// RETURNS the full matrix as a vector<vector<double>>
vector<vector<double>> sparseRow_to_full(vector<int> &start, vector<int> &index, vector<double> &value){
    auto mat_row = start.size() - 1;
    int mat_cols = 0;
    for (int i = 0; i < index.size(); i++){
        if (index[i] > mat_cols){
            mat_cols = index[i];
        }
    }
    vector<vector<double>> matrix(mat_row, vector<double>(mat_cols + 1, 0));
    int row_count = 0;
    for (int j = 0; j < index.size() - 1; j++){
        matrix[row_count][index[j]] = value[j];
        if (j + 1 == start[row_count + 1]){
            row_count++;
        }
    }
    matrix[mat_row - 1][index.back()] = value.back();
    return matrix;
}

// A function to take a column spare matrix and make it into a full matrix
// RETURNS the full matrix as a vector<vector<double>>
vector<vector<double>> sparseColumn_to_full(vector<int> &start, vector<int> &index, vector<double> &value){
    auto mat_cols = start.size() - 1;
    int mat_row = 0;
    for (int i = 0; i < index.size(); i++){
        if (index[i] > mat_row){
            mat_row = index[i];
        }
    }
    vector<vector<double>> matrix(mat_row + 1, vector<double>(mat_cols, 0));
    int col_count = 0;
    for (int j = 0; j < index.size() - 1; j++)
    {
        matrix[index[j]][col_count] = value[j];
        if (j + 1 == start[col_count + 1]){
            col_count++;
        }
    }
    matrix[index.back()][mat_cols - 1] = value.back();
    return matrix;
}






