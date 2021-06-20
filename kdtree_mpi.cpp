#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <limits>
#include "Utility.hpp"

#define DEBUG 0

/* rank: holds the rank of the process
 * size: Tells us how many processes are running parallely*/
int rank, size;

/***************************************************************************************/
float Point::distance_squared(Point &a, Point &b){
    if(a.dimension != b.dimension){
        std::cout << "Dimensions do not match!" << std::endl;
        exit(1);
    }
    float dist = 0;
    for(int i = 0; i < a.dimension; ++i){
        float tmp = a.coordinates[i] - b.coordinates[i];
        dist += tmp * tmp;
    }
    return dist;
}
/***************************************************************************************/


/***************************************************************************************/
Node* build_tree_rec(Point** point_list, int num_points, int depth){
    if (num_points <= 0){
        return nullptr;
    }

    if (num_points == 1){
        return new Node(point_list[0], nullptr, nullptr);
    }

    int dim = point_list[0]->dimension;

    // sort list of points based on axis
    int axis = depth % dim;
    using std::placeholders::_1;
    using std::placeholders::_2;

    std::sort(
            point_list, point_list + (num_points - 1),
            std::bind(Point::compare, _1, _2, axis));

    // select median
    Point** median = point_list + (num_points / 2);
    Point** left_points = point_list;
    Point** right_points = median + 1;

    int num_points_left = num_points / 2;
    int num_points_right = num_points - (num_points / 2) - 1;

    // left subtree
    Node* left_node = build_tree_rec(left_points, num_points_left, depth + 1);

    // right subtree
    Node* right_node = build_tree_rec(right_points, num_points_right, depth + 1);

    // return median node
    return new Node(*median, left_node, right_node);
}

Node* build_tree(Point** point_list, int num_nodes){
    return build_tree_rec(point_list, num_nodes, 0);
}
/***************************************************************************************/


/***************************************************************************************/
Node* nearest(Node* root, Point* query, int depth, Node* best, float &best_dist) {
    // leaf node
    if (root == nullptr){
        return nullptr;
    }

    int dim = query->dimension;
    int axis = depth % dim;

    Node* best_local = best;
    float best_dist_local = best_dist;

    float d_euclidian = root->point->distance_squared(*query);
    float d_axis = query->coordinates[axis] - root->point->coordinates[axis];
    float d_axis_squared = d_axis * d_axis;

    if (d_euclidian < best_dist_local){
        best_local = root;
        best_dist_local = d_euclidian;
    }

    Node* visit_branch;
    Node* other_branch;

    if(d_axis < 0){
        // query point is smaller than root node in axis dimension, i.e. go left
        visit_branch = root->left;
        other_branch = root->right;
    } else{
        // query point is larger than root node in axis dimension, i.e. go right
        visit_branch = root->right;
        other_branch = root->left;
    }

    Node* further = nearest(visit_branch, query, depth + 1, best_local, best_dist_local);
    if (further != nullptr){
        float dist_further = further->point->distance_squared(*query);
        if (dist_further < best_dist_local){
            best_dist_local = dist_further;
            best_local = further;
        }
    }

    if (d_axis_squared < best_dist_local) {
        further = nearest(other_branch, query, depth + 1, best_local, best_dist_local);
        if (further != nullptr){
            float dist_further = further->point->distance_squared(*query);
            if (dist_further < best_dist_local){
                // best_dist_local = dist_further;
                best_local = further;
            }
        }
    }

    return best_local;
}


Node* nearest_neighbor(Node* root, Point* query){
    float best_dist = root->point->distance_squared(*query);
    return nearest(root, query, 0, root, best_dist);
}


/***************************************************************************************/
int main(int argc, char **argv){
    int seed = 0;
    int dim = 0;
    int num_points = 0;
    int num_queries = 10;

    /*Initialize the MPI computation*/
    MPI_Init(&argc, &argv);

    /*Read the rank and size of the processes into rank and size variables*/
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::chrono::high_resolution_clock::time_point tick;

    /* Our approach involves process with rank 0 acting as the main process and the rest of the 15 processes will be the worker processes
     * If the rank of the process is 0(our main process), we read the seed, dim and num_points*/
    if (rank == 0) {
#if DEBUG
        // for measuring your local runtime
        tick = std::chrono::high_resolution_clock::now();
        Utility::specify_problem(argc, argv, &seed, &dim, &num_points);

#else
        Utility::specify_problem(&seed, &dim, &num_points);

#endif

    }

    /* As only the process with rank 0 has the seed, dim and num_points, we broadcast these to all other worker processes.
     * These will be needed by other processes to geenerate the Points*/
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // last points are query
    float* x = Utility::generate_problem(seed, dim, num_points + num_queries);

    /*Variable to hold minimum distance for the query points*/
    float global_min_dist = 0;

    /* chunk_size defines the amount of work that each worker process needs to do.*/
    int chunk_size = std::ceil(num_points / (size-1));

    /* Define the functionality of the main process.
     * Main process(rank 0) will send each worker process with the information on what each worker process needs to do.
     * start and end define the portion of the data each worker process need to work on.
     * After allocating the work to each worker, results are printed by the main process*/
    if(rank == 0) {
        int qid = -999;
        float local_min_distance = std::numeric_limits<float>::max();
        int start = 0;
        int end = std::min(start + chunk_size, num_points);
        for (int i = 1; i < size; i++) {
            MPI_Send(&start, 1 , MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&end, 1 , MPI_INT, i, 1, MPI_COMM_WORLD);
            start = end;
            end = start + chunk_size;
        }
        for (int q = 0; q < num_queries; q++) {
            for (int i = 1; i < size; i++) {
                MPI_Recv(&qid, 1 , MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            }
            MPI_Reduce(&local_min_distance, &global_min_dist, 1 , MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
            Utility::print_result_line(qid, global_min_dist);
        }


#if DEBUG
        // for measuring your local runtime
        auto tock = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = tock - tick;
        std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
#endif

        std::cout << "DONE" << std::endl;

    }
    /* Defines the functionality of each worker.
     * Each worker process receives start and end from the main process. start and end decides what portion of data each worker can work on.
     * Using this, each worker builds a subtree of points. For each query, each worker will find a local nearest neighbor and the minimum distance to that neighbor
     * MPI_Reduce with MPI_MIN operator is used to find the minimum distance for each query across all workers.*/
    else {
        //code for process with other ranks
        float local_min_distance = 0;
        int start, end;
        MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        Point** points = (Point**)calloc(chunk_size, sizeof(Point*));
        int i = 0;
        for(int n = start; n < end; ++n){
            points[i] = new Point(dim, n + 1, x + n * dim);
            i++;

        }

        assert(i == chunk_size);
        // build tree
        Node* tree = build_tree(points, chunk_size);
        // for each query, find nearest neighbor
        for(int q = 0; q < num_queries; ++q){
            float* x_query = x + (num_points + q) * dim;
            Point query(dim, num_points + q, x_query);

            Node* res = nearest_neighbor(tree, &query);

            // output min-distance (i.e. to query point)
            local_min_distance = query.distance(*res->point);
            MPI_Send(&query.ID, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Reduce(&local_min_distance, &global_min_dist, 1 , MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
            //Utility::print_result_line(query.ID, global_min_dist);

#if DEBUG
            // in case you want to have further debug information about
            // the query point and the nearest neighbor
            // std::cout << "Query: " << query << std::endl;
            // std::cout << "NN: " << *res->point << std::endl << std::endl;
#endif
        }

        Utility::free_tree(tree);

        for(int n = 0; n < chunk_size; ++n){
            delete points[n];
        }
        free(points);


    }


    free(x);

    (void)argc;
    (void)argv;
    MPI_Finalize();
    return 0;
}
/***************************************************************************************/