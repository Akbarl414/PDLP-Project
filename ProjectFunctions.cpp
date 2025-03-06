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
void vectorPrint(vector<int> &arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        cout << arr[i] << "\t";
    }
    cout << endl;
}

void PDLP::assignLpValues(int &num_row, int &num_col, int &num_nonZero, vector<double> &cost,
    vector<double> &bound, vector<double> &lp_matrix_values, vector<int> &lp_matrix_index, 
    vector<int> &lp_matrix_start, string modelName, string filePathName)
{
    num_rows = num_row;
    num_cols = num_col;
    num_nonZeros = num_nonZero;
    costs = cost;
    bounds = bound;
    matrix_values = lp_matrix_values;
    matrix_index = lp_matrix_index;
    matrix_start = lp_matrix_start;
    scaled_matrix_values.resize(num_nonZeros);
    x_k.assign(num_cols, 0);
    x_k1.assign(num_cols, 0);
    y_k.assign(num_rows, 0);
    y_k1.assign(num_rows, 0);
    reducedCosts.assign(num_cols, 0);
    feasibility_tolerance = 10E-5;  // Make it 10E-4 or 10E-8 (which is the square of 10E-4) 
    iter_cap = 2.5E6;
    model = modelName; 
    filePath = filePathName;
    // matrix_norm = matrixNorm();
    // step_size = 1/matrix_norm;
    // initialiseNorms();
}

void PDLP::run_Rescale(){
    if(alternate_Scaling){
        ruiz_Rescale_alternate();
        if(chamPockStatus) chamPock_Rescale_alternate();
    }
    else{
        ruiz_Rescale();
        if(chamPockStatus) chamPock_Rescale();
    }
    scaleLP();
}

void PDLP::initialiseModel(){
    if(step_size == 0){
        matrix_norm = matrixNorm();
        step_size = 1/matrix_norm;
    }
    initialiseNorms();
    primal_feasibility_tolerance = feasibility_tolerance * (1 + bounds_2_norm);
    dual_feasibility_tolerance  = feasibility_tolerance * (1 + costs_2_norm);
    primal_step_size = step_size/step_balance;
    dual_step_size = step_size * step_balance;
    if(step_balance == 0 || flatten_step_size){ 
        dual_step_size = step_size; 
        primal_step_size = step_size;
    }
    printf("The dual_feas_tolerance = %g, and prim_feas_tol = %g, dual_s_s = %g and prim_s_s = %g. \n", dual_feasibility_tolerance, primal_feasibility_tolerance,dual_step_size, primal_step_size);
}



void PDLP::initialiseNorms(){
    double bounds_squared_err = 0; 
    double costs_squared_err = 0; 
    for(int iCol = 0; iCol < num_cols; iCol++){
        costs_squared_err += pow(costs[iCol], 2);
    }
    for(int iRow = 0; iRow < num_rows; iRow++){
        bounds_squared_err += pow(bounds[iRow],2);
        // printf("%g \t", bounds[iRow]);
    }
    bounds_2_norm = sqrt(bounds_squared_err);
    costs_2_norm = sqrt(costs_squared_err); 
    step_balance = costs_2_norm/bounds_2_norm;
    printf("Bounds norm is %g \n", bounds_2_norm);
    assert(bounds_2_norm > 10E-16);
    // assert(costs_2_norm > 10E-6);
}


double PDLP::matrixNorm()
{
    vector<double> xk, w, z;
    xk.assign(num_cols, 1);
    w.resize(num_cols);
    int iter = 0;
    double w_norm = 0;
    double dl_x_norm = 0;
    for (;;)
    {
        // Form z = Ax_k
        //double x_norm =0;
        z.assign(num_rows, 0);
        for (int iCol = 0; iCol < num_cols; iCol++){
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){    
                z[matrix_index[iEl]] += matrix_values[iEl] * xk[iCol];
            }
        }
        
        for (int iCol = 0; iCol < num_cols; iCol++){
            w[iCol] = 0;
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++){  
                  w[iCol] += matrix_values[iEl] * z[matrix_index[iEl]];
            }
        }
        
        w_norm = 0;
        for (int iCol = 0; iCol < num_cols; iCol++){
            w_norm = max(abs(w[iCol]), w_norm);
        }
        assert(w_norm > 0);
        dl_x_norm = 0;
        for (int iCol = 0; iCol < num_cols; iCol++){
            w[iCol] /= w_norm;
            dl_x_norm = max(abs(w[iCol] - xk[iCol]), dl_x_norm);
        }
        xk = w;
        // This was used when debugging
        // printf("Iteration %4d: w_norm = %g; dl_x_norm = %g\n", int(iter), w_norm, dl_x_norm);
        if (iter > 1000 || dl_x_norm < 1e-10)
            break;
        iter++;
    }
    //printf("Iteration %4d: w_norm = %g; dl_x_norm = %g\n", int(iter), w_norm, dl_x_norm);
    if(debugFlag) printf("||A||_2 = %g\n", sqrt(w_norm));
    return sqrt(w_norm);
}

void PDLP::PDHGUpdate()
{
    vector<double> AtYk(num_cols, 0);
    
    for (int iCol = 0; iCol < num_cols; iCol++){
        double value = 0;
        for (int columnK = matrix_start[iCol]; columnK < matrix_start[iCol + 1]; columnK++){
            int iRow = matrix_index[columnK];
            value += matrix_values[columnK] * y_k[iRow];
        }
        AtYk[iCol] = value;
    }

    for (int iCol = 0; iCol < num_cols; iCol++){
        x_k1[iCol] = ((x_k[iCol] + primal_step_size * AtYk[iCol] - primal_step_size * costs[iCol]));
        if (x_k1[iCol] < 0){
            x_k1[iCol] = 0;
        }
    }

    // Similarly to how we did for x_k+1 we need to do the matrix-vector multiplication
    // first to solve s*A*(x_k+1 - x_k)
    vector<double> s_Ax(num_rows, 0);
    
    for (int iCol = 0; iCol < num_cols; iCol++)
    {
        for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++)
        {
            s_Ax[matrix_index[column]] += dual_step_size * matrix_values[column] * (2 * x_k1[iCol] - x_k[iCol]);
        }
    }

    for (int iRow = 0; iRow < num_rows; iRow++)
    {
        y_k1[iRow] = y_k[iRow] - s_Ax[iRow] + dual_step_size * bounds[iRow];
    }

}

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

// double PDLP::primal_weight_norm(vector<double> &x, vector<double> &x_hash, vector<double> &y, vector<double> &y_hash ){
    
// }   


vector<double> PDLP::vectorSubtraction(vector<double> &vect1, vector<double> &vect2)
{
    assert(vect1.size() == vect2.size());
    vector<double> result(vect1.size(), 0);
    for (int index = 0; index < vect1.size(); index++)
    {
        result[index] = vect1[index] - vect2[index];
    }
    return result;
}

double PDLP::vectorEuclidianNorm(vector<double> &vect)
{
    double norm_squared = 0;
    for (int index = 0; index < vect.size(); index++)
    {
        norm_squared += pow(vect[index], 2);
    }
    return sqrt(norm_squared);
}  

bool PDLP::updateCriteria()
{
    vector<double> x_norm_inside = vector_Subtraction(x_k1, x_k);
    vector<double> y_norm_inside = vector_Subtraction(y_k1, y_k);

    double update_criteria_x = vectorNorm(x_norm_inside);
    double update_criteria_y = vectorNorm(y_norm_inside);

    if (update_criteria_x > 0.00001 && update_criteria_y > 0.00001)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

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
void PDLP::runFeasiblePDHG(bool outputFlag, bool debugFlagValue)
{
    debugFlag = debugFlagValue; 
    // printf("DebugFlag is %i \n", debugFlag);
    // if (outputFlag == 1) printf("Running Resatarted PDHG with a step size of %g \n", step_size);
    up = 0;
    iterations = 0;
    // PDHGUpdate();
    // restartSolve();

    // up = isFeasible();
    // while(iterations<20){
    while(!up){
        //vectorPrint(x_k);
        PDHGUpdate();
        // printf("we are comparing the values %g and %g",1 - pow((iterations +1),-0.3)*adaptive_step(), (1+pow(iterations+1, -0.6)*step_size));
        // step_size = min((1 - pow((iterations + 1),-0.3))*adaptive_step(),
        //                 (1+pow(iterations+1, -0.6))*step_size);
        // printf("step size is%g \n",step_size);
        restartSolve();
        // addResults();
        up = isFeasible();
        iterations ++;
    }
    
    if(outputFlag ==1) {
        // printf("Restarted PDHG ran with %i iterations \n", iterations);
        if(statusRescale){
            printf("For the scaled model we have: \n");
            isComplementarity();
            printObjectiveValue();
        }
    }
    if(statusRescale){
        unScaleLP();
        calculateReducedCosts();
        isDualFeasible();
        isPrimalFeasible();
        isDualityGap();
        isComplementarity();
    } 
    getObjectiveValue();
    
    printf("In the end we have: \n");
}

void PDLP::addResults(){
    for(int iCol = 0; iCol < num_cols; iCol ++){
        x_k_result.push_back(x_k[iCol]);
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        y_k_result.push_back(y_k[iRow]);
    }
}
void PDLP::writeFile(double &runTime){
    fstream MyFile;
    MyFile.open(filePath + ".csv", ios::app);

    // MyFile << "Model, Obj, Iterations, Alt_Scaling, Duality Gap, Complementarity, Dual feasibility, Primal relative feasibility, num_rows, num_cols, num_nonZeros, Dual_feas_tolerance, Primal_feas_tolerance, Run Time (s) \n";
    MyFile << model.c_str() << ","<< objectiveValue<< ","<<iterations << "," << alternate_Scaling << "," << dualityGap<< "," << adjusted_complementarity << ","<<dual_inf_norm<< ","<<primal_relative_error<< "," << num_rows<< ","<< num_cols<<"," << num_nonZeros <<"," << dual_feasibility_tolerance <<"," << primal_feasibility_tolerance<< "," << runTime/1000<<"," << endl;
    MyFile.close();

    // fstream ResultFile;
    // ResultFile.open("Lu24_xy.csv", ios::app);
    // ResultFile << "X_values" << "," << "Y_values" << "," << "Iteration" << "," << endl;
    // for(auto index = 0; index < x_k_result.size(); index ++){
    //     ResultFile << x_k_result[index] << "," << y_k_result[index] << "," << index << "," <<endl;
    // }
    // ResultFile.close();
}

void PDLP::getObjectiveValue()
{
    double resultValue; 
    for (int iCols = 0; iCols < num_cols; iCols++)
        {
            resultValue += costs[iCols] * x_k[iCols];
        }
    objectiveValue = resultValue;
}

void PDLP::printObjectiveValue(){
    getObjectiveValue();
    if(!up) cout << "Make sure you have Run the model \n"; 
    else
    {
        // printf("Obj, Iterations, adjusted_Complmentarity, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g \n", objectiveValue, iterations, adjusted_complementarity, dual_inf_norm, primal_relative_error);
        printf("Model, Obj, Iterations, Duality Gap, Complementarity, Dual feasibility, and Primal relative feasibility is: %s, %g, %i, %g, %g, %g, %g \n", model.c_str(), objectiveValue, iterations, dualityGap, adjusted_complementarity, dual_inf_norm, primal_relative_error);
        //printf("Our optimal objective value is: %g \n", objectiveValue);
    }   
}

void PDLP::printFullResults()
{
    printf("Our optimal objective value is: %g \n", objectiveValue);
    cout << "Optimal Primal values \n";
    for (double values : x_k)
    {
        printf("%g \t", round(values));
    }
    cout << endl;
    cout << "Optimal Dual values" << endl;
    for (double values : y_k)
    {
        printf("%g \t", round(values));
    }
    cout << endl;
}

void PDLP::restartSolve()
{
    for (int i = 0; i < num_cols; i++){
        x_k[i] = x_k1[i];
        x_k1[i] = 0;
    }
    for (int j = 0; j < num_rows; j++){
        y_k[j] = y_k1[j];
        y_k1[j] = 0;
    }
    primal_step_size = step_size/step_balance;
    dual_step_size = step_size * step_balance;
}

void PDLP::calculateReducedCosts()
{
    vector<double> AtYt(num_cols, 0);
    for (int iCol = 0; iCol < num_cols; iCol++){
        double value = 0;
        for (int columnK = matrix_start[iCol]; columnK < matrix_start[iCol + 1]; columnK++){
            int iRow = matrix_index[columnK];
            value += matrix_values[columnK] * y_k[iRow];
        }
        AtYt[iCol] = value;
    }
    for(int iCol = 0; iCol < num_cols; iCol++){
        double reducedCosti = costs[iCol] - AtYt[iCol];
        if(-reducedCosti < 0){
            reducedCosts[iCol] = reducedCosti; 
        }
        else reducedCosts[iCol] = 0;
    }
}

bool PDLP::isPrimalFeasible(){
    /************************
     A function that returns a Bool value
     It returns true if the primal feasibility is met
     That is if Ax - b < relative primal tolerance 
     ************************/
    double absolute_error;
    double bound_square_error = 0;
    primal_relative_error = 0;
    vector<double> Ax(num_rows, 0);
    for (int iCol = 0; iCol < num_cols; iCol++){
        for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++){
            Ax[matrix_index[column]] += matrix_values[column] * x_k[iCol];
        }
    }
    // Check primal feasibility (absolute and relative norms)
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
bool PDLP::isDualFeasible(){
    /************************
     A function that returns a Bool value
     It returns true if the Dual feasibility is met
     That is if |c - A^Ty - \lamda| < relative dual tolerance 
     ************************/
    dual_inf_norm = 0;
    double dual_feasibility = 0; 
    vector<double> AtYt(num_cols, 0);
    for (int iCol = 0; iCol < num_cols; iCol++){
        double value = 0;
        for (int columnK = matrix_start[iCol]; columnK < matrix_start[iCol + 1]; columnK++){
            int iRow = matrix_index[columnK];
            value += matrix_values[columnK] * y_k[iRow];
        }
        AtYt[iCol] = value;
    }
    for(int iCol = 0; iCol < num_cols; iCol++ ){       
        dual_feasibility += abs(costs[iCol] - AtYt[iCol] - reducedCosts[iCol]);        
    }
    dual_inf_norm = sqrt(max(dual_feasibility, 10E-20));
    // printf("Dual fesibiltity is %g \n", dual_feasibility);
    if(dual_inf_norm > dual_feasibility_tolerance) return false;
    return true;
}


bool PDLP::isComplementarity(){
    complementarity = 0;
    adjusted_complementarity = 0; 
    for(int iCol = 0; iCol < num_cols; iCol++ ){
        complementarity += x_k[iCol]*abs(reducedCosts[iCol]);
        adjusted_complementarity += x_k[iCol]*abs(reducedCosts[iCol]/costs_2_norm);
    }

    if (adjusted_complementarity < feasibility_tolerance) return true;  
    

    return false; 
}
bool PDLP::isDualityGap(){
    double btY = 0;
    double ctX = 0;
    for(int iCol = 0; iCol < num_cols; iCol++){
        ctX += x_k[iCol] * costs[iCol];
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        btY += bounds[iRow] * y_k[iRow];
    }
    dualityGap = abs(btY - ctX);
    if(dualityGap < feasibility_tolerance*(1 + abs(ctX) + abs(btY))) return true;
    return false;
}
bool PDLP::isFeasible(){
    calculateReducedCosts();
    if(isPrimalFeasible() && isDualityGap() && isDualFeasible()) return true;
    // if(isPrimalFeasible() && isComplementarity() && isDualFeasible()) return true;
    if(debugFlag && iterations%20000 == 0) //Include the norms and completmentarity 
    {
        getObjectiveValue();
        isPrimalFeasible(); isDualityGap(); isDualFeasible(); isComplementarity();
        //printf("After %i iterations, Complementarity is %g, Dual feasibility is %i, Primal absolute(2-norm) feasibility is %g and relative is %g \n", iterations, complementarity, isDualFeasible(), sqrt(primal_2_norm), primal_relative_error);
        //printf("Iterations, Complementarity, Dual feasibility, Primal absolute(2-norm) feasibility, and Primal relative is:  %i, %g, %i, %g , %g \n", iterations, complementarity, isDualFeasible(), (primal_2_norm), primal_relative_error);
        // printf("Obj, Iterations, Complementarity, adjusted_Complmentarity, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g, %g \n", objectiveValue,iterations, complementarity, adjusted_complementarity, dual_inf_norm, primal_relative_error);
        // printf("Obj, Iterations, adjusted_Complmentarity, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g \n", objectiveValue,iterations, adjusted_complementarity, dual_inf_norm, primal_relative_error);
        printf("Obj, Iterations, Duality Gap, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g \n", objectiveValue,iterations, dualityGap, dual_inf_norm, primal_relative_error);
        //vectorPrint(reducedCosts);
    }
    if(iterations > iter_cap) {
        // printf("YOU DIDNT CONVERGE: After %i iterations, Complementarity is %g, Dual feasibility is %i, Primal feasibility is %i \n", iterations, complementarity, isDualFeasible(), isPrimalFeasible());
        // printf("Obj, Iterations, Complementarity, adjusted_Complmentarity, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g, %g \n", objectiveValue,iterations, complementarity, adjusted_complementarity, dual_inf_norm, primal_relative_error);
        printf("Obj, Iterations, Duality Gap, Dual feasibility, and Primal relative feasibility is:  %g, %i, %g, %g, %g \n", objectiveValue,iterations, dualityGap, dual_inf_norm, primal_relative_error);
        printDebug();
        return true;
    }    
    return false;
}

void PDLP::printDebug()
{
    if(debugFlag)
    {
        double absolute_error;
        vector<double> Ax(num_cols, 0);
        for (int iCol = 0; iCol < num_cols; iCol++)
        {
            for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++)
            {
                Ax[matrix_index[column]] += matrix_values[column] * x_k[iCol];
            }
        }

        double two_norm_squared = 0;
        for (int iCol = 0; iCol < num_cols; iCol++)
        {
        absolute_error = abs(Ax[iCol] - bounds[iCol]);
        two_norm_squared += pow(absolute_error, 2);
        primal_relative_error = max(primal_relative_error, absolute_error);
        
        // If either absolute or relative norm is violated, return false
       
        }
        // for(int iCol =0; iCol < num_cols; iCol++)
        // {
        //     if(Ax[iCol] - bounds[iCol] > feasibility_tolerance)
        //         printf("The difference is %g and our culprit is %i \n", (sqrt(two_norm_squared)), iCol);
        // }
        // //vectorPrint(x_k);
    }    
    // else{
    //     printf("Nothing happened.");
    // }
    
}


void PDLP::ruiz_Rescale(){ 
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
            // diag_c[iCol] = sqrt(max(diag_c[iCol],scale_epsilon)); //make diag_c = sqrt(D_C)
            diag_c[iCol] = sqrt(diag_c[iCol]);
            diag_c[iCol] = 1/diag_c[iCol];
        }
        
        for(int iRow = 0 ; iRow < num_rows; iRow ++ ){
            // if(diag_r[iRow] != 1) printf("uh oh diag_r is %g \t", diag_r[iRow]);
            // if(iRow == 526) printf("uh oh diag_r is %g \t", diag_r[iRow]);
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
        if(abs(scaled_matrix_values[index]) > 1.00000000001) printf("Index %i at value %g \t ", index, abs(scaled_matrix_values[index]));
    }
}


void PDLP::scale_Column(vector<double> &scalingVector){
    for(int iCol = 0; iCol < num_cols; iCol ++){
        for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl ++){
            scaled_matrix_values[iEl] = scaled_matrix_values[iEl]*(scalingVector[iCol]); 
        }            
    }

}

void PDLP::scale_Row(vector<double> &scalingVector){
    for(int iCol = 0 ; iCol < num_cols; iCol ++ ){
        for(int iEl = matrix_start[iCol]; iEl < matrix_start[iCol +1]; iEl++){
            double value = scaled_matrix_values[iEl];
            double multiplier = scalingVector[matrix_index[iEl]];
            scaled_matrix_values[iEl] = value*(multiplier);    
        }   
    }
}

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
        //Make the 
        col_norm[iCol] = 1 / sqrt(col_norm[iCol]);
    }
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



void PDLP::scaleLP(){
    //This stuff needs to be rescaled. 
    if(!chamPockStatus){  //If chambolle-pock scaling is not used have its scaling values be 1
        col_norm.assign(num_cols, 1);
        row_norm.assign(num_rows, 1);
    }
    orignal_costs = costs;
    orignal_bounds = bounds;
    orignal_matrix_values = matrix_values;
    matrix_values = scaled_matrix_values; 
    // vectorPrint(row_norm);
    for(int iCol = 0; iCol < num_cols; iCol ++){
        double scale_value = col_norm[iCol]  * costs[iCol];
        scale_value = scale_value * d_c[iCol];
        costs[iCol] = scale_value ; 
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        bounds[iRow] = (d_r[iRow])* (row_norm[iRow]) * bounds[iRow];
    }
}

void PDLP::unScaleLP(){
    post_scaled_x_k.resize(num_cols);
    post_scaled_y_k.resize(num_rows);
    
    for(int iCol = 0; iCol < num_cols; iCol ++){
        post_scaled_x_k[iCol] =  (col_norm[iCol])*(d_c[iCol])*x_k[iCol];
    }
    for(int iRow = 0; iRow < num_rows; iRow ++){
        post_scaled_y_k[iRow] = (row_norm[iRow])*(d_r[iRow])*y_k[iRow];
    }
    bounds = orignal_bounds;
    costs = orignal_costs; 
    x_k = post_scaled_x_k;
    y_k = post_scaled_y_k;
    matrix_values = orignal_matrix_values;

}


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
    
    // for(int index = 0; index < num_nonZeros; index++ ){
    //     // if(abs(scaled_matrix_values[index]-sense_check[index]) > 1E-5) printf("d'Oh Index %i with the difference %g \t ", index, abs(scaled_matrix_values[index]-sense_check[index]));
    //     // if(abs(scaled_matrix_values[index]) > 1.00000000001) printf("Index %i at value %g \t ", index, abs(scaled_matrix_values[index]));
    // }
}















// A function that RETURNS the transpose of the inputted matrix
vector<vector<double>> transposeMatrix(vector<vector<double>> &mat)
{
    vector<vector<double>> trans(mat[0].size(), vector<double>(mat.size(), 0));
    for (int i = 0; i < trans.size(); i++)
    {
        for (auto j = 0; j < trans[0].size(); j++)
        {
            trans[i][j] = mat[j][i];
        }
    }
    return trans;
}








// A function that runs the update for the PDHG increments
// Its inputs are x_k and y_k the current values of x and y; x_k1 & y_k1 the next values of x & y, the coefficient matrix A, stepsize s, and vectors b &c
void PDHGupdate(vector<double> &x_k, vector<double> &y_k, vector<double> &x_k1, vector<double> &y_k1, double &s, vector<vector<double>> &A, vector<double> &b, vector<double> &c)
{
    // Initialize the size of the x vector and the matrix A
    auto arrSize = x_k.size();
    auto matRow = A.size();

    // To compute the update for x we will need to first find A^T*y_k
    vector<double> AtYk(arrSize, 0);
    for (int i = 0; i < matRow; i++)
    {
        for (int j = 0; j < arrSize; j++)
        {
            AtYk[j] += A[i][j] * y_k[i];
        }
    }
    // for(auto i : AtYk){
    //     printf("%g \t", i);
    // }
    // cout << endl;
    // Now the update for the x_k+1 where x_k+1 = proj_\R+(x_k + s*A^T*y_k - s*c)
    for (int i = 0; i < arrSize; i++)
    {
        x_k1[i] = (x_k[i] + s * AtYk[i] - s * c[i]);
        if (x_k1[i] < 0)
        {
            x_k1[i] = 0;
        }
    }

    // Similarly to how we did for x_k+1 we need to do the matrix-vector multiplication first to solve s*A*(x_k+1 - x_k)
    vector<double> s_Ax(arrSize, 0);
    for (int i = 0; i < matRow; i++)
    {
        for (int j = 0; j < arrSize; j++)
        {
            s_Ax[i] += s * A[i][j] * (2 * x_k1[j] - x_k[j]);
        }
    }
    // cout << "Dense s_Ax values: \n";
    //  for (double p : s_Ax){
    //     printf("%g \t", p);
    // }
    // cout << endl;
    // printf("The size of Dense s_Ax is %i \n", int(s_Ax.size()));
    for (int i = 0; i < y_k.size(); i++)
    {
        y_k1[i] = y_k[i] - s_Ax[i] + s * b[i];
    }
}

void PDHGupdate_sparse(vector<double> &x_k, vector<double> &y_k, vector<double> &x_k1,
                       vector<double> &y_k1, double &s, vector<double> &a_values, vector<int> &a_start,
                       vector<int> &a_index, vector<double> &b, vector<double> &c)
{
    // Initialize the size of the x vector and the matrix A
    auto arrSize = x_k.size();
    auto matRow = y_k.size();
    // double precision = pow(10, 11);
    // printf("precision %g \n", precision);
    //  To compute the update for x we will need to first find A^T*y_k
    vector<double> AtYk(arrSize, 0);
    for (int j = 0; j < (a_start.size() - 1); j++)
    {
        double value = 0;
        for (int k = a_start[j]; k < a_start[j + 1]; k++)
        {
            int iRow = a_index[k];
            value += a_values[k] * y_k[iRow];
        }
        AtYk[j] = value;
    }

    // for(auto i : AtYk){
    //     printf("%g \t", i);
    // }

    for (int i = 0; i < arrSize; i++)
    {
        x_k1[i] = ((x_k[i] + s * AtYk[i] - s * c[i]));
        if (x_k1[i] < 0)
        {
            x_k1[i] = 0;
        }
    }

    // Similarly to how we did for x_k+1 we need to do the matrix-vector multiplication first to solve s*A*(x_k+1 - x_k)
    vector<double> s_Ax(arrSize, 0);
    // for (int i  = 0; i < arrSize; i++){
    //     for (int j = 0; j < (a_index.size() - 1); j++){
    //        if(a_index[j] == i){
    //         s_Ax[i] += a_values[j];
    //        }

    //     }
    // }
    for (int i = 0; i < a_start.size() - 1; i++)
    {
        for (int el = a_start[i]; el < a_start[i + 1]; el++)
        {
            s_Ax[a_index[el]] += s * a_values[el] * (2 * x_k1[i] - x_k[i]);
        }
    }
    // for (double p : s_Ax){
    //     printf("%g \t", p);
    // }
    // printf("The size of Sparse s_Ax is %i \n", int(s_Ax.size()));
    for (int i = 0; i < y_k.size(); i++)
    {
        y_k1[i] = y_k[i] - s_Ax[i] + s * b[i];
    }
}

// A function to calculate the Euclidian norm ||n||_2 of any inputted vector
double vectorNorm(vector<double> &vector)
{
    double norm_squared = 0;
    for (int i = 0; i < vector.size(); i++)
    {
        norm_squared += pow(vector[i], 2);
    }
    return sqrt(norm_squared);
}

// A function that returns the difference of two input vectors
vector<double> vector_Subtraction(vector<double> &vect1, vector<double> &vect2)
{
    if (vect1.size() != vect2.size())
    {
        return vector<double>(2, 0);
    }
    vector<double> result(vect1.size(), 0);
    for (int i = 0; i < vect1.size(); i++)
    {
        result[i] = vect1[i] - vect2[i];
    }
    return result;
}

// A function that returns a boolean of whether or not the update criteria (norm of (x_k1 - x_k) < 0.01)
bool stop_update(vector<double> &x_k, vector<double> &x_k1, vector<double> &y_k, vector<double> &y_k1)
{
    vector<double> x_norm_inside = vector_Subtraction(x_k1, x_k);
    vector<double> y_norm_inside = vector_Subtraction(y_k1, y_k);

    double update_criteria_x = vectorNorm(x_norm_inside);
    double update_criteria_y = vectorNorm(y_norm_inside);

    if (update_criteria_x > 0.00001 && update_criteria_y > 0.00001)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

void restart_solve(vector<double> &x_k, vector<double> &x_k1, vector<double> &y_k, vector<double> &y_k1)
{
    for (int i = 0; i < x_k.size(); i++)
    {
        x_k[i] = x_k1[i];
        x_k1[i] = 0;
    }
    for (int j = 0; j < y_k.size(); j++)
    {
        y_k[j] = y_k1[j];
        y_k1[j] = 0;
    }
}

// A function that RETURNS the product of the inputted matricies, given their dimensions fit
vector<vector<double>> matrixMult(vector<vector<double>> &mat1, vector<vector<double>> &mat2)
{
    vector<vector<double>> result(mat1.size(), vector<double>(mat2[0].size(), 0));
    for (int i = 0; i < mat1.size(); i++)
    {
        for (int j = 0; j < mat2[0].size(); j++)
        {
            for (int k = 0; k < mat1[0].size(); k++)
            {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

// Boolean to check if it a square matrix
bool isSquareMatrix(vector<vector<double>> &mat)
{
    if (mat[0].size() != mat.size())
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

double sparseMatrixNorm(vector<double> &value, vector<int> &start, vector<int> &index, int &num_row, int &num_col)
{
    vector<double> xk, w, z;
    xk.assign(num_col, 1);
    w.resize(num_col);
    int iter = 0;
    double w_norm = 0;
    double dl_x_norm = 0;
    for (;;)
    {
        // Form z = Ax_k
        z.assign(num_row, 0);
        for (int iCol = 0; iCol < num_col; iCol++)
        {
            for (int iEl = start[iCol]; iEl < start[iCol + 1]; iEl++)
                z[index[iEl]] += value[iEl] * xk[iCol];
        }
        // Form w = A^Tz
        //
        for (int iCol = 0; iCol < num_col; iCol++)
        {
            w[iCol] = 0;
            for (int iEl = start[iCol]; iEl < start[iCol + 1]; iEl++)
                w[iCol] += value[iEl] * z[index[iEl]];
        }
        // Normalise w
        w_norm = 0;
        for (int iCol = 0; iCol < num_col; iCol++)
        {
            w_norm += max(abs(w[iCol]), w_norm);
            // printf("%g \t", w_norm);
        }
        assert(w_norm > 0);
        dl_x_norm = 0;
        for (int iCol = 0; iCol < num_col; iCol++)
        {
            w[iCol] /= w_norm;
            dl_x_norm += max(abs(w[iCol] - xk[iCol]), dl_x_norm);
        }
        xk = w;
        // This was used when debugging
        //
        // printf("Iteration %4d: w_norm = %g; dl_x_norm = %g\n", int(iter), w_norm, dl_x_norm);
        if (iter > 1000 || dl_x_norm < 1e-10)
            break;
        iter++;
    }
    printf("Iteration %4d: w_norm = %g; dl_x_norm = %g\n", int(iter), w_norm, dl_x_norm);
    printf("||A||_2 = %g\n", sqrt(w_norm));
    return sqrt(w_norm);
}

// Now a function to calculate the ||A||_2 value of a matrix
double matrixNorm(vector<vector<double>> &mat)
{
    int num_row = int(mat.size());
    int num_col = int(mat[0].size());
    vector<double> xk, w, z;
    xk.assign(num_col, 1);
    w.resize(num_col);
    int iter = 0;
    double w_norm = 0;
    double dl_x_norm = 0;
    for (;;)
    {
        // printf("iteration: %i, num rows: %i, num col: %i \n", iter, num_row, num_col);
        //  Form z = Ax_k
        z.assign(num_row, 0);
        for (int iCol = 0; iCol < num_col; iCol++)
        {
            for (int j = 0; j < num_row; j++)
                z[j] += mat[j][iCol] * xk[iCol];
        }
        // Form w = A^Tz
        //
        for (int iCol = 0; iCol < num_col; iCol++)
        {
            w[iCol] = 0;
            for (int j = 0; j < num_row; j++)
                w[iCol] += mat[j][iCol] * z[j];
        }
        // Normalise w
        w_norm = 0;
        for (int iCol = 0; iCol < num_col; iCol++)
            w_norm += max(abs(w[iCol]), w_norm);
        assert(w_norm > 0);
        dl_x_norm = 0;
        for (int iCol = 0; iCol < num_col; iCol++)
        {
            w[iCol] /= w_norm;
            dl_x_norm += max(abs(w[iCol] - xk[iCol]), dl_x_norm);
        }
        xk = w;
        // This was used when debugging
        //
        // printf("Iteration %4d: w_norm = %g; dl_x_norm = %g\n", int(iter), w_norm, dl_x_norm);
        if (iter > 1000 || dl_x_norm < 1e-10)
            break;
        iter++;
    }
    //    printf("Iteration %4d: w_norm = %g; dl_x_norm = %g\n", iter, w_norm, dl_x_norm);
    //    printf("||A||_2 = %g\n", sqrt(w_norm));
    return sqrt(w_norm);
}

// A function to take a row spare matrix and make it into a full matrix
// RETURNS the full matrix as a vector<vector<double>>
vector<vector<double>> sparseRow_to_full(vector<int> &start, vector<int> &index, vector<double> &value)
{
    auto mat_row = start.size() - 1;
    int mat_cols = 0;
    for (int i = 0; i < index.size(); i++)
    {
        if (index[i] > mat_cols)
        {
            mat_cols = index[i];
        }
    }
    vector<vector<double>> matrix(mat_row, vector<double>(mat_cols + 1, 0));
    int row_count = 0;
    for (int j = 0; j < index.size() - 1; j++)
    {
        matrix[row_count][index[j]] = value[j];
        if (j + 1 == start[row_count + 1])
        {
            row_count++;
        }
    }
    matrix[mat_row - 1][index.back()] = value.back();
    return matrix;
}

// A function to take a column spare matrix and make it into a full matrix
// RETURNS the full matrix as a vector<vector<double>>
vector<vector<double>> sparseColumn_to_full(vector<int> &start, vector<int> &index, vector<double> &value)
{
    auto mat_cols = start.size() - 1;
    int mat_row = 0;
    for (int i = 0; i < index.size(); i++)
    {
        if (index[i] > mat_row)
        {
            mat_row = index[i];
        }
    }
    vector<vector<double>> matrix(mat_row + 1, vector<double>(mat_cols, 0));
    int col_count = 0;
    for (int j = 0; j < index.size() - 1; j++)
    {
        matrix[index[j]][col_count] = value[j];
        if (j + 1 == start[col_count + 1])
        {
            col_count++;
        }
    }
    matrix[index.back()][mat_cols - 1] = value.back();
    return matrix;
}
