//
//  ProjectFunctions.hpp
//  MathsProjectCode
//
//  Created by Akbar Latif on 10/18/24.
//
#ifndef ProjectFunctions_hpp
#define ProjectFunctions_hpp

#include <stdio.h>
#include <vector>
using namespace std;

void matPrint(vector<vector<double>> &mat);                                                                                                                                        // Declaration
void vectorPrint(vector<double> &arr);                                                                                                                                             // Declaration
vector<vector<double>> transposeMatrix(vector<vector<double>> &mat);                                                                                                               // Declaration
bool sizeMatch(vector<vector<double>> &mat, vector<double> &arr);                                                                                                                  // Declaration
vector<double> matrixArrayMult(vector<vector<double>> &mat, vector<double> &arr);                                                                                                  // Declaration
void PDHGupdate(vector<double> &x_k, vector<double> &y_k, vector<double> &x_k1, vector<double> &y_k1, double &s, vector<vector<double>> &A, vector<double> &b, vector<double> &c); // Declaration
vector<double> matMult_middleterm(vector<vector<double>> &mat, vector<double> &arr, double &s);                                                                                    // Declaration

double vectorNorm(vector<double> &vect);                                                                  // Declaration
vector<double> vector_Subtraction(vector<double> &vect1, vector<double> &vect2);                          // Declaration
bool stop_update(vector<double> &x_k, vector<double> &x_k1, vector<double> &y_k, vector<double> &y_k1);   // Declaration
void restart_solve(vector<double> &x_k, vector<double> &x_k1, vector<double> &y_k, vector<double> &y_k1); // Declaration
vector<vector<double>> matrixMult(vector<vector<double>> &mat1, vector<vector<double>> &mat2);            // Declaration
// vector<vector<double>> matrixNorm(vector<vector<double>> &mat);//Declaration
bool isSquareMatrix(vector<vector<double>> &mat);          // Declaration
void round_vector(vector<double> &vec, double &precision); // Declaration
double round_to(double &value, double &precision);
double sparseMatrixNorm(vector<double> &value, vector<int> &start, vector<int> &index, int &num_row, int &num_col);
double matrixNorm(vector<vector<double>> &mat);
vector<vector<double>> sparseRow_to_full(vector<int> &start, vector<int> &index, vector<double> &value);    // Declaration
vector<vector<double>> sparseColumn_to_full(vector<int> &start, vector<int> &index, vector<double> &value); // Declaration
void PDHGupdate_sparse(vector<double> &x_k, vector<double> &y_k, vector<double> &x_k1,
                       vector<double> &y_k1, double &s, vector<double> &a_values, vector<int> &a_start,
                       vector<int> &a_index, vector<double> &b, vector<double> &c); //Declaration

void printPrimalDual(vector<double> &x_k, vector<double> &y_k);
void printObjectiveValue(vector<double> &x_k, vector<double> &c);
void printAllResults(vector<double> &x_k, vector<double> &y_k, vector<double> &c);


#endif                                                                                                      /* ProjectFunctions_hpp */
