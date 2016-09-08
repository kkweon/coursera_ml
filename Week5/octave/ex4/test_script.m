## Copyright (C) 2016 bnlmath
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} test_script (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: bnlmath <bnlmath@bnlmath-ubuntu>
## Created: 2016-09-08

function [J grad] = test_script (lambda)
il = 2;              % input layer
hl = 2;              % hidden layer
nl = 4;              % number of labels
nn = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];
lambda = lambda;
[J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda);

endfunction
