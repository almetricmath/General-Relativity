This project provides python code to go along with the TensorTransform writeup. There are driver programs for 2nd, 3rd, and 4th order tensors. Each order tensor has a driver program that computes the tensor by outer product and inner product in the unprimed coordinate system. It also changes the coordinate system and computes the outer product and inner product formulations of a tensor in the primed coordinate system. There are also drivers to test all the configurations - raising and lowering all the indices using the metric and inverse metric in both the unprimed and primed coordinate systems.

The output of the python code is in a latex format compatible with MS Word - See (https://support.microsoft.com/en-us/office/linear-format-equations-using-unicodemath-and-latex-in-word-2e00618d-b1fd-49d8-8cb4-8d17f25754f8). The latex used in MS Word is not standard latex. The output can be pasted into Ms Word equation editor in latex mode and it will render.

The program has been run in ubuntu linux and on a Windows terminal - see ( https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701?hl=en-us&gl=us&rtc=1). The programs output subscripts and superscripts that work in these two programs. In the Windows command prompt, some of the subscripts and superscipts don't render properly.

In the fourth order drivers, the outer product routine does not output the full sum because it's too large for MS Word to render - it appears to lock up. It outputs each element of the sum and the result.

The blockDotTensorProductInterchange.py file shows an example of W.[E x T] =  [E] x [W.T] where  '.' is a block dot product and 'x' is a tensor product - see discussion in write up.
