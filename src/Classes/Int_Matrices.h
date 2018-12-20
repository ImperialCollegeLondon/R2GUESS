/* This file is part of GUESS.
 *      Copyright (c) Marc Chadeau-Hyam (m.chadeau@imperial.ac.uk)
 *                    Leonardo Bottolo (l.bottolo@imperial.ac.uk)
 *                    David Hastie (d.hastie@imperial.ac.uk)
 *      2010
 *
 * GUESS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GUESS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GUESS.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef INT_MATRICES_H
#define INT_MATRICES_H 1

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

class Int_Matrices
{
public:
  Int_Matrices();
  ~Int_Matrices(){};

  int nb_rows;
  int nb_columns;
  int **matrix;
  

  void Alloc_int_matrix(int Rows,
			   int Columns);
  void Replicate_int_matrix(Int_Matrices Source);
  void Free_int_matrix();
  void Read_from_file(char *filename);
  void Display_matrix();
  void Display_matrix_header();
  void Write_to_file(char *filename);

};

#endif /* !defined INT_MATRICES_H */
