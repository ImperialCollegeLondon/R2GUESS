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

#ifndef DYN_NAME_H
#define DYN_NAME_H

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "struc.h"

using namespace std;

string Get_dyn_name(string File_name,
		    int Number,
		    string Extension);

void Write_dyn_double(data_double * object,
		      char * File_name,
		      string Name_number1 ,
		      int Number1,
		      string Name_number2 ,
		      int Number2,
		      string Extension);



void Write_dyn_double_bis(data_double * object,
			  char * File_name,
			  string Name_number1 ,
			  int Number1,
			  string Name_number2 ,
			  int Number2,
			  string Name_number3 ,
			  int Number3,
			  string Extension);

string Get_stddzed_name(string File_name,
			int Number,
			string Name_number,
			string Output_type,
			string Extension);
string Get_stddzed_name_short(string File_name,
			      string Output_type,
			      string Extension);
void Write_matrix_double(data_double *M, char *file_name);

#endif /* DYN_NAME_H */
