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
// #include Highs.h

using namespace std;

void Rectangle::printDimensions()
{
    printf("The width of the rectangle is %g and the length is %g \n", width, length);
}

void Rectangle::initWhy(bool &value){
    why = value;
}

void Rectangle::printWhy(){
    printf("Why is %d \n", why);
    changeWhy();
}

void Rectangle::changeWhy(){
    if (why != 0) why = 0;
    else why = 1;
}
// A function that prints the input matrix of any size when called
void matPrint(vector<vector<double>> &mat)
{
    for (int i = 0; i < mat.size(); i++)
    {
        for (auto j = 0; j < mat[0].size(); j++)
        {
            cout << mat[i][j] << "\t";
        }
        cout << endl;
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

void PDLP::assignLpValues(int &num_row, int &num_col, int &num_nonZero, vector<double> &cost,
    vector<double> &bound, vector<double> &lp_matrix_values, vector<int> &lp_matrix_index, 
    vector<int> &lp_matrix_start)
{
    num_rows = num_row;
    num_cols = num_col;
    num_nonZeros = num_nonZero;
    costs = cost;
    bounds = bound;
    matrix_values = lp_matrix_values;
    matrix_index = lp_matrix_index;
    matrix_start = lp_matrix_start;
    matrix_norm = matrixNorm();
    x_k.assign(num_cols, 0);
    x_k1.assign(num_cols, 0);
    y_k.assign(num_rows, 0);
    y_k1.assign(num_rows, 0);
    step_size = 1/matrix_norm;
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
        z.assign(num_rows, 0);
        for (int iCol = 0; iCol < num_cols; iCol++)
        {
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++)
                z[matrix_index[iEl]] += matrix_values[iEl] * xk[iCol];
        }
        // Form w = A^Tz
        //
        for (int iCol = 0; iCol < num_cols; iCol++)
        {
            w[iCol] = 0;
            for (int iEl = matrix_start[iCol]; iEl < matrix_start[iCol + 1]; iEl++)
                w[iCol] += matrix_values[iEl] * z[matrix_index[iEl]];
        }
        // Normalise w
        w_norm = 0;
        for (int iCol = 0; iCol < num_cols; iCol++)
        {
            w_norm += max(abs(w[iCol]), w_norm);
            // printf("%g \t", w_norm);
        }
        assert(w_norm > 0);
        dl_x_norm = 0;
        for (int iCol = 0; iCol < num_cols; iCol++)
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

void PDLP::PDHGUpdate()
{
    vector<double> AtYk(num_cols, 0);
    for (int iCol = 0; iCol < num_cols; iCol++)
    {
        double value = 0;
        for (int columnK = matrix_start[iCol]; columnK < matrix_start[iCol + 1]; columnK++)
        {
            int iRow = matrix_index[columnK];
            value += matrix_values[columnK] * y_k[iRow];
        }
        AtYk[iCol] = value;
    }

    for (int iCol = 0; iCol < num_cols; iCol++)
    {
        x_k1[iCol] = ((x_k[iCol] + step_size * AtYk[iCol] - step_size * costs[iCol]));
        if (x_k1[iCol] < 0)
        {
            x_k1[iCol] = 0;
        }
    }

    // Similarly to how we did for x_k+1 we need to do the matrix-vector multiplication
    // first to solve s*A*(x_k+1 - x_k)
    vector<double> s_Ax(num_cols, 0);
    
    for (int iCol = 0; iCol < num_cols; iCol++)
    {
        for (int column = matrix_start[iCol]; column < matrix_start[iCol + 1]; column++)
        {
            s_Ax[matrix_index[column]] += step_size * matrix_values[column] * (2 * x_k1[iCol] - x_k[iCol]);
        }
    }

    for (int iRow = 0; iRow < num_rows; iRow++)
    {
        y_k1[iRow] = y_k[iRow] - s_Ax[iRow] + step_size * bounds[iRow];
    }

}

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

void PDLP::printObjectiveValue()
{
    if(!up) cout << "Make sure you have Run the model \n"; 
    else
    {
        printf("Our optimal objective value is: %g \n", objectiveValue);
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
    for (int i = 0; i < num_cols; i++)
    {
        x_k[i] = x_k1[i];
        x_k1[i] = 0;
    }
    for (int j = 0; j < num_rows; j++)
    {
        y_k[j] = y_k1[j];
        y_k1[j] = 0;
    }
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

// RETURNS a boolean that verifies size of your vector and matrix line up so they can be multiplied together
bool sizeMatch(vector<vector<double>> &mat, vector<double> &arr)
{
    if (mat[0].size() != arr.size())
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

// A function that RETURNS the product of the inputted matrix and vector, given their dimensions fit, if not it returns (0,0)
vector<double> matrixArrayMult(vector<vector<double>> &mat, vector<double> &arr)
{
    // cout << "check size match" << sizeMatch(mat, arr) << endl;
    assert(sizeMatch(mat, arr));
    vector<double> result(mat.size(), 0);
    for (int i = 0; i < mat.size(); i++)
    {
        for (int j = 0; j < mat[0].size(); j++)
        {
            result[i] += mat[i][j] * arr[j];
        }
    }
    return result;
}

// A functions that RETURNS the product of the inputted matrix, vector, and double
vector<double> matMult_middleterm(vector<vector<double>> &mat, vector<double> &arr, double &s)
{
    vector<double> result(matrixArrayMult(mat, arr));
    for (int i = 0; i < result.size(); i++)
    {
        result[i] = result[i] * s;
    }
    return result;
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

void printPrimalDual(vector<double> &x_k, vector<double> &y_k)
{
    cout << "Optimal x \n";
    for (double i : x_k)
    {
        printf("%g \t", round(i));
    }
    cout << endl;

    cout << "Optimal y" << endl;
    vectorPrint(y_k);
}

void printObjectiveValue(vector<double> &x_k, vector<double> &c)
{
    double objectiveValue;
    for (int i = 0; i < c.size(); i++)
    {
        objectiveValue += c[i] * x_k[i];
    }
    printf("Our optimal objective value is: %g \n", objectiveValue);
}

void printAllResults(vector<double> &x_k, vector<double> &y_k, vector<double> &c)
{
    double objectiveValue;
    for (int i = 0; i < c.size(); i++)
    {
        objectiveValue += c[i] * x_k[i];
    }
    printf("Our optimal objective value is: %g \n", objectiveValue);
    cout << "Optimal x \n";
    for (double i : x_k)
    {
        printf("%g \t", round(i));
    }
    cout << endl;

    cout << "Optimal y" << endl;
    vectorPrint(y_k);
}

// void passHighs(vector<double> &matrix_values, vector<int> &vector_start, vector<int> &vector_index, vector<int> &costs, vector<int> &bounds, Highs h){
//     h.getLP().a.matrix_
// }

// double  round_to(double &value, double &precision){
//     return round(value/precision)*(precision);
// }

// void round_vector(vector<double> &vec, double &precision ){
//     //vector<double> result(vec.size(),0);
//     for(int i = 0; i < vec.size(); i++){
//         vec[i] = round(vec[i]/precision)*(precision);
//     }
//     //return result;
//
// }
