functions{

	vector cross(vector a, vector b){
		/*Return the cross product of vectors a and b*/

		vector[3] output;

		output[1] = a[2] * b[3] - a[3] * b[2];
		output[2] = a[3] * b[1] - a[1] * b[3];
		output[3] = a[1] * b[2] - a[2] * b[1];
	    
	    return output;

	}


	vector rotate_rodrig(vector v, vector k){
		/*Rotate vector v around vector k.
		  The length of k encodes the magnitude of the rotation*/

		real theta;

		theta = sqrt(dot_product(k,k));
		return v * cos(theta) + cross(k,v) * sin(theta) + k * dot_product(k, v) * (1-cos(theta));

	}


	matrix position_object(matrix board, vector rod, vector tran){
		/*position a chessboard in the world by rotatiting it with rod and translating with tran */

		matrix[rows(board), cols(board)] positioned_board;

		for (point in 1:rows(board)){

			positioned_board[point] = (rotate_rodrig(board[point]', rod) + tran)';

		}

		return positioned_board;


	}

	matrix project_object(matrix board, matrix M){
		/*Project an object onto the camera using matrix M*/

		matrix[rows(board), cols(board)] projected_image_homog;
		matrix[rows(board), cols(board)] projected_image;

		for (point in 1:rows(board)){

			projected_image_homog[point] = (M * board[point]')';

			projected_image[point] = projected_image_homog[point] / projected_image_homog[point, 3];

		}

		return projected_image;

	}

}

data {
	
	int J; //Number of points in chessboard
	matrix[J, 3] chessboard;

	matrix[3,3] M;

	vector[3] rod;
	vector[3] trans;

}


parameters{
	

}


model{
	

}


generated quantities{
	
	matrix[J, 3] new_object;

	matrix[J, 3] projection;

	new_object = position_object(chessboard, rod, trans);

	projection = project_object(new_object, M);

}