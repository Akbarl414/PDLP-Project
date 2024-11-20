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
        //vectorPrint(snx_k);
        // printf("The code performed %d iterations \n", iter);
        // printObjectiveValue(snx_k, c);
        // for(int i = 0; i < c.size(); i++){
        // if (c[i] != 0){
        //     printf("We have for column %i our HiGHS solution gives %g where PDHG gives us %g at cost %g \n", i, solution.col_value[i], snx_k[i],c[i]);
        // }
    }
    
    else
    {
        cout << "Try again didn't specify your model properly \n";
    }

    printf("The code performed %d iterations \n", iter);
    printObjectiveValue(snx_k, c);
    //printAllResults(x_k, y_k, c);
    //cout << "Vector costs are:" << endl;
    //vectorPrint(c);
    for(int i = 0; i < c.size(); i++){
        if (c[i] != 0){
            printf("We have for column %i our HiGHS solution gives %g where PDHG gives us %g at cost %g \n", i, solution.col_value[i], snx_k[i],c[i]);
        }
    }


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
//     cout << "Dense" << endl;
//     PDHGupdate(x_k, y_k, x_k1, y_k1, s, A, b, c);
//     cout << "x_k's : \n";
//     vectorPrint(x_k1);
//     cout << "y_k's : \n";
//     vectorPrint(y_k1);
//     cout << y_k1.size() << endl;

//     // vector<double> dense_x_k = x_k1;
//     // vector<double> dense_y_k = y_k1;
//     // x_k.assign(num_cols, 0);
//     // y_k.assign(num_rows, 0);
//     // x_k1.assign(x_k.size(), 0);
//     // y_k1.assign(y_k.size(), 0);

//    vector<double> sx_k(num_cols, 0);
//     vector<double> sy_k(num_rows, 0);

//     vector<double> sx_k1(x_k.size(), 0);
//     vector<double> sy_k1(y_k.size(), 0);
   
   
//     cout << "Sparse" << endl;
//     PDHGupdate_sparse(sx_k, sy_k, sx_k1, sy_k1, s, A_value, A_start, A_index, b, c);
//     cout << "sparse x_k's : \n";
//     vectorPrint(sx_k1);
//     cout << "sparse y_k's : \n";
//     vectorPrint(sy_k1);
//     //cout << y_k1.size() << endl;
//     for(int i =0; i < y_k1.size(); i++){
//         if(sy_k1[i] != y_k1[i]){
//             //printf("dOh! %g does not equal %g \n", dense_y_k[i],  y_k1[i]);

//             printf("dOh! %g does not equal %g and the difference is %g \n", sy_k1[i],  y_k1[i], abs(sy_k1[i] - y_k1[i]));
//         }
//     }
//     assert(sx_k1 == x_k1);
//     assert(sy_k1 == y_k1);
