��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
�
Adam/OutputLayer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/OutputLayer/bias/v

+Adam/OutputLayer/bias/v/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/bias/v*
_output_shapes
:*
dtype0
�
Adam/OutputLayer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#**
shared_nameAdam/OutputLayer/kernel/v
�
-Adam/OutputLayer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/kernel/v*
_output_shapes

:#*
dtype0
�
Adam/HiddenLayer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*)
shared_nameAdam/HiddenLayer3/bias/v
�
,Adam/HiddenLayer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer3/bias/v*
_output_shapes
:#*
dtype0
�
Adam/HiddenLayer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:##*+
shared_nameAdam/HiddenLayer3/kernel/v
�
.Adam/HiddenLayer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer3/kernel/v*
_output_shapes

:##*
dtype0
�
Adam/HiddenLayer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*)
shared_nameAdam/HiddenLayer2/bias/v
�
,Adam/HiddenLayer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer2/bias/v*
_output_shapes
:#*
dtype0
�
Adam/HiddenLayer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:##*+
shared_nameAdam/HiddenLayer2/kernel/v
�
.Adam/HiddenLayer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer2/kernel/v*
_output_shapes

:##*
dtype0
�
Adam/HiddenLayer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*)
shared_nameAdam/HiddenLayer1/bias/v
�
,Adam/HiddenLayer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer1/bias/v*
_output_shapes
:#*
dtype0
�
Adam/HiddenLayer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*+
shared_nameAdam/HiddenLayer1/kernel/v
�
.Adam/HiddenLayer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer1/kernel/v*
_output_shapes

:#*
dtype0
�
Adam/OutputLayer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/OutputLayer/bias/m

+Adam/OutputLayer/bias/m/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/bias/m*
_output_shapes
:*
dtype0
�
Adam/OutputLayer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#**
shared_nameAdam/OutputLayer/kernel/m
�
-Adam/OutputLayer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/kernel/m*
_output_shapes

:#*
dtype0
�
Adam/HiddenLayer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*)
shared_nameAdam/HiddenLayer3/bias/m
�
,Adam/HiddenLayer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer3/bias/m*
_output_shapes
:#*
dtype0
�
Adam/HiddenLayer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:##*+
shared_nameAdam/HiddenLayer3/kernel/m
�
.Adam/HiddenLayer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer3/kernel/m*
_output_shapes

:##*
dtype0
�
Adam/HiddenLayer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*)
shared_nameAdam/HiddenLayer2/bias/m
�
,Adam/HiddenLayer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer2/bias/m*
_output_shapes
:#*
dtype0
�
Adam/HiddenLayer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:##*+
shared_nameAdam/HiddenLayer2/kernel/m
�
.Adam/HiddenLayer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer2/kernel/m*
_output_shapes

:##*
dtype0
�
Adam/HiddenLayer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*)
shared_nameAdam/HiddenLayer1/bias/m
�
,Adam/HiddenLayer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer1/bias/m*
_output_shapes
:#*
dtype0
�
Adam/HiddenLayer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*+
shared_nameAdam/HiddenLayer1/kernel/m
�
.Adam/HiddenLayer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HiddenLayer1/kernel/m*
_output_shapes

:#*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
x
OutputLayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameOutputLayer/bias
q
$OutputLayer/bias/Read/ReadVariableOpReadVariableOpOutputLayer/bias*
_output_shapes
:*
dtype0
�
OutputLayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*#
shared_nameOutputLayer/kernel
y
&OutputLayer/kernel/Read/ReadVariableOpReadVariableOpOutputLayer/kernel*
_output_shapes

:#*
dtype0
z
HiddenLayer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*"
shared_nameHiddenLayer3/bias
s
%HiddenLayer3/bias/Read/ReadVariableOpReadVariableOpHiddenLayer3/bias*
_output_shapes
:#*
dtype0
�
HiddenLayer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:##*$
shared_nameHiddenLayer3/kernel
{
'HiddenLayer3/kernel/Read/ReadVariableOpReadVariableOpHiddenLayer3/kernel*
_output_shapes

:##*
dtype0
z
HiddenLayer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*"
shared_nameHiddenLayer2/bias
s
%HiddenLayer2/bias/Read/ReadVariableOpReadVariableOpHiddenLayer2/bias*
_output_shapes
:#*
dtype0
�
HiddenLayer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:##*$
shared_nameHiddenLayer2/kernel
{
'HiddenLayer2/kernel/Read/ReadVariableOpReadVariableOpHiddenLayer2/kernel*
_output_shapes

:##*
dtype0
z
HiddenLayer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*"
shared_nameHiddenLayer1/bias
s
%HiddenLayer1/bias/Read/ReadVariableOpReadVariableOpHiddenLayer1/bias*
_output_shapes
:#*
dtype0
�
HiddenLayer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*$
shared_nameHiddenLayer1/kernel
{
'HiddenLayer1/kernel/Read/ReadVariableOpReadVariableOpHiddenLayer1/kernel*
_output_shapes

:#*
dtype0
}
serving_default_InputLayerPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputLayerHiddenLayer1/kernelHiddenLayer1/biasHiddenLayer2/kernelHiddenLayer2/biasHiddenLayer3/kernelHiddenLayer3/biasOutputLayer/kernelOutputLayer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_414603

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�: B�:
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
<
0
1
2
3
%4
&5
-6
.7*
<
0
1
2
3
%4
&5
-6
.7*
* 
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
4trace_0
5trace_1
6trace_2
7trace_3* 
6
8trace_0
9trace_1
:trace_2
;trace_3* 
* 
�
<iter

=beta_1

>beta_2
	?decay
@learning_ratemimjmkml%mm&mn-mo.mpvqvrvsvt%vu&vv-vw.vx*

Aserving_default* 

0
1*

0
1*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 
c]
VARIABLE_VALUEHiddenLayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEHiddenLayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
c]
VARIABLE_VALUEHiddenLayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEHiddenLayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Utrace_0* 

Vtrace_0* 
c]
VARIABLE_VALUEHiddenLayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEHiddenLayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

\trace_0* 

]trace_0* 
b\
VARIABLE_VALUEOutputLayer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEOutputLayer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

^0
_1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
`	variables
a	keras_api
	btotal
	ccount*
H
d	variables
e	keras_api
	ftotal
	gcount
h
_fn_kwargs*

b0
c1*

`	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

f0
g1*

d	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
��
VARIABLE_VALUEAdam/HiddenLayer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/HiddenLayer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/HiddenLayer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/HiddenLayer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/HiddenLayer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/HiddenLayer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/OutputLayer/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/OutputLayer/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/HiddenLayer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/HiddenLayer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/HiddenLayer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/HiddenLayer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/HiddenLayer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/HiddenLayer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/OutputLayer/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/OutputLayer/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'HiddenLayer1/kernel/Read/ReadVariableOp%HiddenLayer1/bias/Read/ReadVariableOp'HiddenLayer2/kernel/Read/ReadVariableOp%HiddenLayer2/bias/Read/ReadVariableOp'HiddenLayer3/kernel/Read/ReadVariableOp%HiddenLayer3/bias/Read/ReadVariableOp&OutputLayer/kernel/Read/ReadVariableOp$OutputLayer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/HiddenLayer1/kernel/m/Read/ReadVariableOp,Adam/HiddenLayer1/bias/m/Read/ReadVariableOp.Adam/HiddenLayer2/kernel/m/Read/ReadVariableOp,Adam/HiddenLayer2/bias/m/Read/ReadVariableOp.Adam/HiddenLayer3/kernel/m/Read/ReadVariableOp,Adam/HiddenLayer3/bias/m/Read/ReadVariableOp-Adam/OutputLayer/kernel/m/Read/ReadVariableOp+Adam/OutputLayer/bias/m/Read/ReadVariableOp.Adam/HiddenLayer1/kernel/v/Read/ReadVariableOp,Adam/HiddenLayer1/bias/v/Read/ReadVariableOp.Adam/HiddenLayer2/kernel/v/Read/ReadVariableOp,Adam/HiddenLayer2/bias/v/Read/ReadVariableOp.Adam/HiddenLayer3/kernel/v/Read/ReadVariableOp,Adam/HiddenLayer3/bias/v/Read/ReadVariableOp-Adam/OutputLayer/kernel/v/Read/ReadVariableOp+Adam/OutputLayer/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_414908
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHiddenLayer1/kernelHiddenLayer1/biasHiddenLayer2/kernelHiddenLayer2/biasHiddenLayer3/kernelHiddenLayer3/biasOutputLayer/kernelOutputLayer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/HiddenLayer1/kernel/mAdam/HiddenLayer1/bias/mAdam/HiddenLayer2/kernel/mAdam/HiddenLayer2/bias/mAdam/HiddenLayer3/kernel/mAdam/HiddenLayer3/bias/mAdam/OutputLayer/kernel/mAdam/OutputLayer/bias/mAdam/HiddenLayer1/kernel/vAdam/HiddenLayer1/bias/vAdam/HiddenLayer2/kernel/vAdam/HiddenLayer2/bias/vAdam/HiddenLayer3/kernel/vAdam/HiddenLayer3/bias/vAdam/OutputLayer/kernel/vAdam/OutputLayer/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_415017��
�&
�
D__inference_NN_model_layer_call_and_return_conditional_losses_414707

inputs=
+hiddenlayer1_matmul_readvariableop_resource:#:
,hiddenlayer1_biasadd_readvariableop_resource:#=
+hiddenlayer2_matmul_readvariableop_resource:##:
,hiddenlayer2_biasadd_readvariableop_resource:#=
+hiddenlayer3_matmul_readvariableop_resource:##:
,hiddenlayer3_biasadd_readvariableop_resource:#<
*outputlayer_matmul_readvariableop_resource:#9
+outputlayer_biasadd_readvariableop_resource:
identity��#HiddenLayer1/BiasAdd/ReadVariableOp�"HiddenLayer1/MatMul/ReadVariableOp�#HiddenLayer2/BiasAdd/ReadVariableOp�"HiddenLayer2/MatMul/ReadVariableOp�#HiddenLayer3/BiasAdd/ReadVariableOp�"HiddenLayer3/MatMul/ReadVariableOp�"OutputLayer/BiasAdd/ReadVariableOp�!OutputLayer/MatMul/ReadVariableOp�
"HiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0�
HiddenLayer1/MatMulMatMulinputs*HiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
#HiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
HiddenLayer1/BiasAddBiasAddHiddenLayer1/MatMul:product:0+HiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#j
HiddenLayer1/TanhTanhHiddenLayer1/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
"HiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource*
_output_shapes

:##*
dtype0�
HiddenLayer2/MatMulMatMulHiddenLayer1/Tanh:y:0*HiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
#HiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
HiddenLayer2/BiasAddBiasAddHiddenLayer2/MatMul:product:0+HiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#j
HiddenLayer2/TanhTanhHiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
"HiddenLayer3/MatMul/ReadVariableOpReadVariableOp+hiddenlayer3_matmul_readvariableop_resource*
_output_shapes

:##*
dtype0�
HiddenLayer3/MatMulMatMulHiddenLayer2/Tanh:y:0*HiddenLayer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
#HiddenLayer3/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
HiddenLayer3/BiasAddBiasAddHiddenLayer3/MatMul:product:0+HiddenLayer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#j
HiddenLayer3/TanhTanhHiddenLayer3/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
!OutputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0�
OutputLayer/MatMulMatMulHiddenLayer3/Tanh:y:0)OutputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"OutputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
OutputLayer/BiasAddBiasAddOutputLayer/MatMul:product:0*OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������k
IdentityIdentityOutputLayer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^HiddenLayer1/BiasAdd/ReadVariableOp#^HiddenLayer1/MatMul/ReadVariableOp$^HiddenLayer2/BiasAdd/ReadVariableOp#^HiddenLayer2/MatMul/ReadVariableOp$^HiddenLayer3/BiasAdd/ReadVariableOp#^HiddenLayer3/MatMul/ReadVariableOp#^OutputLayer/BiasAdd/ReadVariableOp"^OutputLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2J
#HiddenLayer1/BiasAdd/ReadVariableOp#HiddenLayer1/BiasAdd/ReadVariableOp2H
"HiddenLayer1/MatMul/ReadVariableOp"HiddenLayer1/MatMul/ReadVariableOp2J
#HiddenLayer2/BiasAdd/ReadVariableOp#HiddenLayer2/BiasAdd/ReadVariableOp2H
"HiddenLayer2/MatMul/ReadVariableOp"HiddenLayer2/MatMul/ReadVariableOp2J
#HiddenLayer3/BiasAdd/ReadVariableOp#HiddenLayer3/BiasAdd/ReadVariableOp2H
"HiddenLayer3/MatMul/ReadVariableOp"HiddenLayer3/MatMul/ReadVariableOp2H
"OutputLayer/BiasAdd/ReadVariableOp"OutputLayer/BiasAdd/ReadVariableOp2F
!OutputLayer/MatMul/ReadVariableOp!OutputLayer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_HiddenLayer3_layer_call_fn_414756

inputs
unknown:##
	unknown_0:#
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414357o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������#`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�

�
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414767

inputs0
matmul_readvariableop_resource:##-
biasadd_readvariableop_resource:#
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:##*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������#W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������#w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�
�
-__inference_HiddenLayer2_layer_call_fn_414736

inputs
unknown:##
	unknown_0:#
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414340o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������#`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�	
�
)__inference_NN_model_layer_call_fn_414399

inputlayer
unknown:#
	unknown_0:#
	unknown_1:##
	unknown_2:#
	unknown_3:##
	unknown_4:#
	unknown_5:#
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_NN_model_layer_call_and_return_conditional_losses_414380o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
InputLayer
�

�
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414340

inputs0
matmul_readvariableop_resource:##-
biasadd_readvariableop_resource:#
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:##*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������#W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������#w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�	
�
)__inference_NN_model_layer_call_fn_414645

inputs
unknown:#
	unknown_0:#
	unknown_1:##
	unknown_2:#
	unknown_3:##
	unknown_4:#
	unknown_5:#
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_NN_model_layer_call_and_return_conditional_losses_414486o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
$__inference_signature_wrapper_414603

inputlayer
unknown:#
	unknown_0:#
	unknown_1:##
	unknown_2:#
	unknown_3:##
	unknown_4:#
	unknown_5:#
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_414305o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
InputLayer
�G
�
__inference__traced_save_414908
file_prefix2
.savev2_hiddenlayer1_kernel_read_readvariableop0
,savev2_hiddenlayer1_bias_read_readvariableop2
.savev2_hiddenlayer2_kernel_read_readvariableop0
,savev2_hiddenlayer2_bias_read_readvariableop2
.savev2_hiddenlayer3_kernel_read_readvariableop0
,savev2_hiddenlayer3_bias_read_readvariableop1
-savev2_outputlayer_kernel_read_readvariableop/
+savev2_outputlayer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_hiddenlayer1_kernel_m_read_readvariableop7
3savev2_adam_hiddenlayer1_bias_m_read_readvariableop9
5savev2_adam_hiddenlayer2_kernel_m_read_readvariableop7
3savev2_adam_hiddenlayer2_bias_m_read_readvariableop9
5savev2_adam_hiddenlayer3_kernel_m_read_readvariableop7
3savev2_adam_hiddenlayer3_bias_m_read_readvariableop8
4savev2_adam_outputlayer_kernel_m_read_readvariableop6
2savev2_adam_outputlayer_bias_m_read_readvariableop9
5savev2_adam_hiddenlayer1_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer1_bias_v_read_readvariableop9
5savev2_adam_hiddenlayer2_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer2_bias_v_read_readvariableop9
5savev2_adam_hiddenlayer3_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer3_bias_v_read_readvariableop8
4savev2_adam_outputlayer_kernel_v_read_readvariableop6
2savev2_adam_outputlayer_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddenlayer1_kernel_read_readvariableop,savev2_hiddenlayer1_bias_read_readvariableop.savev2_hiddenlayer2_kernel_read_readvariableop,savev2_hiddenlayer2_bias_read_readvariableop.savev2_hiddenlayer3_kernel_read_readvariableop,savev2_hiddenlayer3_bias_read_readvariableop-savev2_outputlayer_kernel_read_readvariableop+savev2_outputlayer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_hiddenlayer1_kernel_m_read_readvariableop3savev2_adam_hiddenlayer1_bias_m_read_readvariableop5savev2_adam_hiddenlayer2_kernel_m_read_readvariableop3savev2_adam_hiddenlayer2_bias_m_read_readvariableop5savev2_adam_hiddenlayer3_kernel_m_read_readvariableop3savev2_adam_hiddenlayer3_bias_m_read_readvariableop4savev2_adam_outputlayer_kernel_m_read_readvariableop2savev2_adam_outputlayer_bias_m_read_readvariableop5savev2_adam_hiddenlayer1_kernel_v_read_readvariableop3savev2_adam_hiddenlayer1_bias_v_read_readvariableop5savev2_adam_hiddenlayer2_kernel_v_read_readvariableop3savev2_adam_hiddenlayer2_bias_v_read_readvariableop5savev2_adam_hiddenlayer3_kernel_v_read_readvariableop3savev2_adam_hiddenlayer3_bias_v_read_readvariableop4savev2_adam_outputlayer_kernel_v_read_readvariableop2savev2_adam_outputlayer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :#:#:##:#:##:#:#:: : : : : : : : : :#:#:##:#:##:#:#::#:#:##:#:##:#:#:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:#: 

_output_shapes
:#:$ 

_output_shapes

:##: 

_output_shapes
:#:$ 

_output_shapes

:##: 

_output_shapes
:#:$ 

_output_shapes

:#: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:#: 

_output_shapes
:#:$ 

_output_shapes

:##: 

_output_shapes
:#:$ 

_output_shapes

:##: 

_output_shapes
:#:$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

:#: 

_output_shapes
:#:$ 

_output_shapes

:##: 

_output_shapes
:#:$ 

_output_shapes

:##: 

_output_shapes
:#:$  

_output_shapes

:#: !

_output_shapes
::"

_output_shapes
: 
�&
�
D__inference_NN_model_layer_call_and_return_conditional_losses_414676

inputs=
+hiddenlayer1_matmul_readvariableop_resource:#:
,hiddenlayer1_biasadd_readvariableop_resource:#=
+hiddenlayer2_matmul_readvariableop_resource:##:
,hiddenlayer2_biasadd_readvariableop_resource:#=
+hiddenlayer3_matmul_readvariableop_resource:##:
,hiddenlayer3_biasadd_readvariableop_resource:#<
*outputlayer_matmul_readvariableop_resource:#9
+outputlayer_biasadd_readvariableop_resource:
identity��#HiddenLayer1/BiasAdd/ReadVariableOp�"HiddenLayer1/MatMul/ReadVariableOp�#HiddenLayer2/BiasAdd/ReadVariableOp�"HiddenLayer2/MatMul/ReadVariableOp�#HiddenLayer3/BiasAdd/ReadVariableOp�"HiddenLayer3/MatMul/ReadVariableOp�"OutputLayer/BiasAdd/ReadVariableOp�!OutputLayer/MatMul/ReadVariableOp�
"HiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0�
HiddenLayer1/MatMulMatMulinputs*HiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
#HiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
HiddenLayer1/BiasAddBiasAddHiddenLayer1/MatMul:product:0+HiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#j
HiddenLayer1/TanhTanhHiddenLayer1/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
"HiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource*
_output_shapes

:##*
dtype0�
HiddenLayer2/MatMulMatMulHiddenLayer1/Tanh:y:0*HiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
#HiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
HiddenLayer2/BiasAddBiasAddHiddenLayer2/MatMul:product:0+HiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#j
HiddenLayer2/TanhTanhHiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
"HiddenLayer3/MatMul/ReadVariableOpReadVariableOp+hiddenlayer3_matmul_readvariableop_resource*
_output_shapes

:##*
dtype0�
HiddenLayer3/MatMulMatMulHiddenLayer2/Tanh:y:0*HiddenLayer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
#HiddenLayer3/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
HiddenLayer3/BiasAddBiasAddHiddenLayer3/MatMul:product:0+HiddenLayer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#j
HiddenLayer3/TanhTanhHiddenLayer3/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
!OutputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0�
OutputLayer/MatMulMatMulHiddenLayer3/Tanh:y:0)OutputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"OutputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
OutputLayer/BiasAddBiasAddOutputLayer/MatMul:product:0*OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������k
IdentityIdentityOutputLayer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^HiddenLayer1/BiasAdd/ReadVariableOp#^HiddenLayer1/MatMul/ReadVariableOp$^HiddenLayer2/BiasAdd/ReadVariableOp#^HiddenLayer2/MatMul/ReadVariableOp$^HiddenLayer3/BiasAdd/ReadVariableOp#^HiddenLayer3/MatMul/ReadVariableOp#^OutputLayer/BiasAdd/ReadVariableOp"^OutputLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2J
#HiddenLayer1/BiasAdd/ReadVariableOp#HiddenLayer1/BiasAdd/ReadVariableOp2H
"HiddenLayer1/MatMul/ReadVariableOp"HiddenLayer1/MatMul/ReadVariableOp2J
#HiddenLayer2/BiasAdd/ReadVariableOp#HiddenLayer2/BiasAdd/ReadVariableOp2H
"HiddenLayer2/MatMul/ReadVariableOp"HiddenLayer2/MatMul/ReadVariableOp2J
#HiddenLayer3/BiasAdd/ReadVariableOp#HiddenLayer3/BiasAdd/ReadVariableOp2H
"HiddenLayer3/MatMul/ReadVariableOp"HiddenLayer3/MatMul/ReadVariableOp2H
"OutputLayer/BiasAdd/ReadVariableOp"OutputLayer/BiasAdd/ReadVariableOp2F
!OutputLayer/MatMul/ReadVariableOp!OutputLayer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_NN_model_layer_call_and_return_conditional_losses_414550

inputlayer%
hiddenlayer1_414529:#!
hiddenlayer1_414531:#%
hiddenlayer2_414534:##!
hiddenlayer2_414536:#%
hiddenlayer3_414539:##!
hiddenlayer3_414541:#$
outputlayer_414544:# 
outputlayer_414546:
identity��$HiddenLayer1/StatefulPartitionedCall�$HiddenLayer2/StatefulPartitionedCall�$HiddenLayer3/StatefulPartitionedCall�#OutputLayer/StatefulPartitionedCall�
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall
inputlayerhiddenlayer1_414529hiddenlayer1_414531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414323�
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_414534hiddenlayer2_414536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414340�
$HiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0hiddenlayer3_414539hiddenlayer3_414541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414357�
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer3/StatefulPartitionedCall:output:0outputlayer_414544outputlayer_414546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414373{
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall%^HiddenLayer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2L
$HiddenLayer3/StatefulPartitionedCall$HiddenLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
InputLayer
�	
�
)__inference_NN_model_layer_call_fn_414624

inputs
unknown:#
	unknown_0:#
	unknown_1:##
	unknown_2:#
	unknown_3:##
	unknown_4:#
	unknown_5:#
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_NN_model_layer_call_and_return_conditional_losses_414380o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414747

inputs0
matmul_readvariableop_resource:##-
biasadd_readvariableop_resource:#
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:##*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������#W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������#w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�
�
-__inference_HiddenLayer1_layer_call_fn_414716

inputs
unknown:#
	unknown_0:#
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414323o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������#`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_415017
file_prefix6
$assignvariableop_hiddenlayer1_kernel:#2
$assignvariableop_1_hiddenlayer1_bias:#8
&assignvariableop_2_hiddenlayer2_kernel:##2
$assignvariableop_3_hiddenlayer2_bias:#8
&assignvariableop_4_hiddenlayer3_kernel:##2
$assignvariableop_5_hiddenlayer3_bias:#7
%assignvariableop_6_outputlayer_kernel:#1
#assignvariableop_7_outputlayer_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: @
.assignvariableop_17_adam_hiddenlayer1_kernel_m:#:
,assignvariableop_18_adam_hiddenlayer1_bias_m:#@
.assignvariableop_19_adam_hiddenlayer2_kernel_m:##:
,assignvariableop_20_adam_hiddenlayer2_bias_m:#@
.assignvariableop_21_adam_hiddenlayer3_kernel_m:##:
,assignvariableop_22_adam_hiddenlayer3_bias_m:#?
-assignvariableop_23_adam_outputlayer_kernel_m:#9
+assignvariableop_24_adam_outputlayer_bias_m:@
.assignvariableop_25_adam_hiddenlayer1_kernel_v:#:
,assignvariableop_26_adam_hiddenlayer1_bias_v:#@
.assignvariableop_27_adam_hiddenlayer2_kernel_v:##:
,assignvariableop_28_adam_hiddenlayer2_bias_v:#@
.assignvariableop_29_adam_hiddenlayer3_kernel_v:##:
,assignvariableop_30_adam_hiddenlayer3_bias_v:#?
-assignvariableop_31_adam_outputlayer_kernel_v:#9
+assignvariableop_32_adam_outputlayer_bias_v:
identity_34��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp$assignvariableop_hiddenlayer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hiddenlayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_hiddenlayer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_hiddenlayer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_hiddenlayer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_hiddenlayer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_outputlayer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_outputlayer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_hiddenlayer1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_hiddenlayer1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_hiddenlayer2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_hiddenlayer2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_hiddenlayer3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_hiddenlayer3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_outputlayer_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_outputlayer_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_hiddenlayer1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_hiddenlayer1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_hiddenlayer2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_hiddenlayer2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_hiddenlayer3_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_hiddenlayer3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_outputlayer_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_outputlayer_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
D__inference_NN_model_layer_call_and_return_conditional_losses_414380

inputs%
hiddenlayer1_414324:#!
hiddenlayer1_414326:#%
hiddenlayer2_414341:##!
hiddenlayer2_414343:#%
hiddenlayer3_414358:##!
hiddenlayer3_414360:#$
outputlayer_414374:# 
outputlayer_414376:
identity��$HiddenLayer1/StatefulPartitionedCall�$HiddenLayer2/StatefulPartitionedCall�$HiddenLayer3/StatefulPartitionedCall�#OutputLayer/StatefulPartitionedCall�
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCallinputshiddenlayer1_414324hiddenlayer1_414326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414323�
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_414341hiddenlayer2_414343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414340�
$HiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0hiddenlayer3_414358hiddenlayer3_414360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414357�
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer3/StatefulPartitionedCall:output:0outputlayer_414374outputlayer_414376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414373{
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall%^HiddenLayer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2L
$HiddenLayer3/StatefulPartitionedCall$HiddenLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_NN_model_layer_call_and_return_conditional_losses_414574

inputlayer%
hiddenlayer1_414553:#!
hiddenlayer1_414555:#%
hiddenlayer2_414558:##!
hiddenlayer2_414560:#%
hiddenlayer3_414563:##!
hiddenlayer3_414565:#$
outputlayer_414568:# 
outputlayer_414570:
identity��$HiddenLayer1/StatefulPartitionedCall�$HiddenLayer2/StatefulPartitionedCall�$HiddenLayer3/StatefulPartitionedCall�#OutputLayer/StatefulPartitionedCall�
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall
inputlayerhiddenlayer1_414553hiddenlayer1_414555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414323�
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_414558hiddenlayer2_414560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414340�
$HiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0hiddenlayer3_414563hiddenlayer3_414565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414357�
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer3/StatefulPartitionedCall:output:0outputlayer_414568outputlayer_414570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414373{
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall%^HiddenLayer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2L
$HiddenLayer3/StatefulPartitionedCall$HiddenLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
InputLayer
�
�
,__inference_OutputLayer_layer_call_fn_414776

inputs
unknown:#
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414373o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�	
�
)__inference_NN_model_layer_call_fn_414526

inputlayer
unknown:#
	unknown_0:#
	unknown_1:##
	unknown_2:#
	unknown_3:##
	unknown_4:#
	unknown_5:#
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_NN_model_layer_call_and_return_conditional_losses_414486o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������
$
_user_specified_name
InputLayer
�
�
D__inference_NN_model_layer_call_and_return_conditional_losses_414486

inputs%
hiddenlayer1_414465:#!
hiddenlayer1_414467:#%
hiddenlayer2_414470:##!
hiddenlayer2_414472:#%
hiddenlayer3_414475:##!
hiddenlayer3_414477:#$
outputlayer_414480:# 
outputlayer_414482:
identity��$HiddenLayer1/StatefulPartitionedCall�$HiddenLayer2/StatefulPartitionedCall�$HiddenLayer3/StatefulPartitionedCall�#OutputLayer/StatefulPartitionedCall�
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCallinputshiddenlayer1_414465hiddenlayer1_414467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414323�
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_414470hiddenlayer2_414472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414340�
$HiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0hiddenlayer3_414475hiddenlayer3_414477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������#*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414357�
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer3/StatefulPartitionedCall:output:0outputlayer_414480outputlayer_414482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414373{
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall%^HiddenLayer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2L
$HiddenLayer3/StatefulPartitionedCall$HiddenLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
!__inference__wrapped_model_414305

inputlayerF
4nn_model_hiddenlayer1_matmul_readvariableop_resource:#C
5nn_model_hiddenlayer1_biasadd_readvariableop_resource:#F
4nn_model_hiddenlayer2_matmul_readvariableop_resource:##C
5nn_model_hiddenlayer2_biasadd_readvariableop_resource:#F
4nn_model_hiddenlayer3_matmul_readvariableop_resource:##C
5nn_model_hiddenlayer3_biasadd_readvariableop_resource:#E
3nn_model_outputlayer_matmul_readvariableop_resource:#B
4nn_model_outputlayer_biasadd_readvariableop_resource:
identity��,NN_model/HiddenLayer1/BiasAdd/ReadVariableOp�+NN_model/HiddenLayer1/MatMul/ReadVariableOp�,NN_model/HiddenLayer2/BiasAdd/ReadVariableOp�+NN_model/HiddenLayer2/MatMul/ReadVariableOp�,NN_model/HiddenLayer3/BiasAdd/ReadVariableOp�+NN_model/HiddenLayer3/MatMul/ReadVariableOp�+NN_model/OutputLayer/BiasAdd/ReadVariableOp�*NN_model/OutputLayer/MatMul/ReadVariableOp�
+NN_model/HiddenLayer1/MatMul/ReadVariableOpReadVariableOp4nn_model_hiddenlayer1_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0�
NN_model/HiddenLayer1/MatMulMatMul
inputlayer3NN_model/HiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
,NN_model/HiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp5nn_model_hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
NN_model/HiddenLayer1/BiasAddBiasAdd&NN_model/HiddenLayer1/MatMul:product:04NN_model/HiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#|
NN_model/HiddenLayer1/TanhTanh&NN_model/HiddenLayer1/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
+NN_model/HiddenLayer2/MatMul/ReadVariableOpReadVariableOp4nn_model_hiddenlayer2_matmul_readvariableop_resource*
_output_shapes

:##*
dtype0�
NN_model/HiddenLayer2/MatMulMatMulNN_model/HiddenLayer1/Tanh:y:03NN_model/HiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
,NN_model/HiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp5nn_model_hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
NN_model/HiddenLayer2/BiasAddBiasAdd&NN_model/HiddenLayer2/MatMul:product:04NN_model/HiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#|
NN_model/HiddenLayer2/TanhTanh&NN_model/HiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
+NN_model/HiddenLayer3/MatMul/ReadVariableOpReadVariableOp4nn_model_hiddenlayer3_matmul_readvariableop_resource*
_output_shapes

:##*
dtype0�
NN_model/HiddenLayer3/MatMulMatMulNN_model/HiddenLayer2/Tanh:y:03NN_model/HiddenLayer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#�
,NN_model/HiddenLayer3/BiasAdd/ReadVariableOpReadVariableOp5nn_model_hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype0�
NN_model/HiddenLayer3/BiasAddBiasAdd&NN_model/HiddenLayer3/MatMul:product:04NN_model/HiddenLayer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#|
NN_model/HiddenLayer3/TanhTanh&NN_model/HiddenLayer3/BiasAdd:output:0*
T0*'
_output_shapes
:���������#�
*NN_model/OutputLayer/MatMul/ReadVariableOpReadVariableOp3nn_model_outputlayer_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0�
NN_model/OutputLayer/MatMulMatMulNN_model/HiddenLayer3/Tanh:y:02NN_model/OutputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+NN_model/OutputLayer/BiasAdd/ReadVariableOpReadVariableOp4nn_model_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
NN_model/OutputLayer/BiasAddBiasAdd%NN_model/OutputLayer/MatMul:product:03NN_model/OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%NN_model/OutputLayer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^NN_model/HiddenLayer1/BiasAdd/ReadVariableOp,^NN_model/HiddenLayer1/MatMul/ReadVariableOp-^NN_model/HiddenLayer2/BiasAdd/ReadVariableOp,^NN_model/HiddenLayer2/MatMul/ReadVariableOp-^NN_model/HiddenLayer3/BiasAdd/ReadVariableOp,^NN_model/HiddenLayer3/MatMul/ReadVariableOp,^NN_model/OutputLayer/BiasAdd/ReadVariableOp+^NN_model/OutputLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2\
,NN_model/HiddenLayer1/BiasAdd/ReadVariableOp,NN_model/HiddenLayer1/BiasAdd/ReadVariableOp2Z
+NN_model/HiddenLayer1/MatMul/ReadVariableOp+NN_model/HiddenLayer1/MatMul/ReadVariableOp2\
,NN_model/HiddenLayer2/BiasAdd/ReadVariableOp,NN_model/HiddenLayer2/BiasAdd/ReadVariableOp2Z
+NN_model/HiddenLayer2/MatMul/ReadVariableOp+NN_model/HiddenLayer2/MatMul/ReadVariableOp2\
,NN_model/HiddenLayer3/BiasAdd/ReadVariableOp,NN_model/HiddenLayer3/BiasAdd/ReadVariableOp2Z
+NN_model/HiddenLayer3/MatMul/ReadVariableOp+NN_model/HiddenLayer3/MatMul/ReadVariableOp2Z
+NN_model/OutputLayer/BiasAdd/ReadVariableOp+NN_model/OutputLayer/BiasAdd/ReadVariableOp2X
*NN_model/OutputLayer/MatMul/ReadVariableOp*NN_model/OutputLayer/MatMul/ReadVariableOp:S O
'
_output_shapes
:���������
$
_user_specified_name
InputLayer
�

�
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414727

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:#
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������#W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������#w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414373

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�

�
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414357

inputs0
matmul_readvariableop_resource:##-
biasadd_readvariableop_resource:#
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:##*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������#W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������#w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�	
�
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414786

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������#
 
_user_specified_nameinputs
�

�
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414323

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:#
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������#P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������#W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������#w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

InputLayer3
serving_default_InputLayer:0���������?
OutputLayer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
X
0
1
2
3
%4
&5
-6
.7"
trackable_list_wrapper
X
0
1
2
3
%4
&5
-6
.7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
4trace_0
5trace_1
6trace_2
7trace_32�
)__inference_NN_model_layer_call_fn_414399
)__inference_NN_model_layer_call_fn_414624
)__inference_NN_model_layer_call_fn_414645
)__inference_NN_model_layer_call_fn_414526�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z4trace_0z5trace_1z6trace_2z7trace_3
�
8trace_0
9trace_1
:trace_2
;trace_32�
D__inference_NN_model_layer_call_and_return_conditional_losses_414676
D__inference_NN_model_layer_call_and_return_conditional_losses_414707
D__inference_NN_model_layer_call_and_return_conditional_losses_414550
D__inference_NN_model_layer_call_and_return_conditional_losses_414574�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z8trace_0z9trace_1z:trace_2z;trace_3
�B�
!__inference__wrapped_model_414305
InputLayer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
<iter

=beta_1

>beta_2
	?decay
@learning_ratemimjmkml%mm&mn-mo.mpvqvrvsvt%vu&vv-vw.vx"
	optimizer
,
Aserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_02�
-__inference_HiddenLayer1_layer_call_fn_414716�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
�
Htrace_02�
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414727�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
%:##2HiddenLayer1/kernel
:#2HiddenLayer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_02�
-__inference_HiddenLayer2_layer_call_fn_414736�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
�
Otrace_02�
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
%:###2HiddenLayer2/kernel
:#2HiddenLayer2/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
Utrace_02�
-__inference_HiddenLayer3_layer_call_fn_414756�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0
�
Vtrace_02�
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414767�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zVtrace_0
%:###2HiddenLayer3/kernel
:#2HiddenLayer3/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
\trace_02�
,__inference_OutputLayer_layer_call_fn_414776�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0
�
]trace_02�
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414786�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z]trace_0
$:"#2OutputLayer/kernel
:2OutputLayer/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_NN_model_layer_call_fn_414399
InputLayer"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_NN_model_layer_call_fn_414624inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_NN_model_layer_call_fn_414645inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_NN_model_layer_call_fn_414526
InputLayer"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_NN_model_layer_call_and_return_conditional_losses_414676inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_NN_model_layer_call_and_return_conditional_losses_414707inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_NN_model_layer_call_and_return_conditional_losses_414550
InputLayer"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_NN_model_layer_call_and_return_conditional_losses_414574
InputLayer"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
$__inference_signature_wrapper_414603
InputLayer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_HiddenLayer1_layer_call_fn_414716inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414727inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_HiddenLayer2_layer_call_fn_414736inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414747inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_HiddenLayer3_layer_call_fn_414756inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414767inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_OutputLayer_layer_call_fn_414776inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414786inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
`	variables
a	keras_api
	btotal
	ccount"
_tf_keras_metric
^
d	variables
e	keras_api
	ftotal
	gcount
h
_fn_kwargs"
_tf_keras_metric
.
b0
c1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
*:(#2Adam/HiddenLayer1/kernel/m
$:"#2Adam/HiddenLayer1/bias/m
*:(##2Adam/HiddenLayer2/kernel/m
$:"#2Adam/HiddenLayer2/bias/m
*:(##2Adam/HiddenLayer3/kernel/m
$:"#2Adam/HiddenLayer3/bias/m
):'#2Adam/OutputLayer/kernel/m
#:!2Adam/OutputLayer/bias/m
*:(#2Adam/HiddenLayer1/kernel/v
$:"#2Adam/HiddenLayer1/bias/v
*:(##2Adam/HiddenLayer2/kernel/v
$:"#2Adam/HiddenLayer2/bias/v
*:(##2Adam/HiddenLayer3/kernel/v
$:"#2Adam/HiddenLayer3/bias/v
):'#2Adam/OutputLayer/kernel/v
#:!2Adam/OutputLayer/bias/v�
H__inference_HiddenLayer1_layer_call_and_return_conditional_losses_414727\/�,
%�"
 �
inputs���������
� "%�"
�
0���������#
� �
-__inference_HiddenLayer1_layer_call_fn_414716O/�,
%�"
 �
inputs���������
� "����������#�
H__inference_HiddenLayer2_layer_call_and_return_conditional_losses_414747\/�,
%�"
 �
inputs���������#
� "%�"
�
0���������#
� �
-__inference_HiddenLayer2_layer_call_fn_414736O/�,
%�"
 �
inputs���������#
� "����������#�
H__inference_HiddenLayer3_layer_call_and_return_conditional_losses_414767\%&/�,
%�"
 �
inputs���������#
� "%�"
�
0���������#
� �
-__inference_HiddenLayer3_layer_call_fn_414756O%&/�,
%�"
 �
inputs���������#
� "����������#�
D__inference_NN_model_layer_call_and_return_conditional_losses_414550n%&-.;�8
1�.
$�!

InputLayer���������
p 

 
� "%�"
�
0���������
� �
D__inference_NN_model_layer_call_and_return_conditional_losses_414574n%&-.;�8
1�.
$�!

InputLayer���������
p

 
� "%�"
�
0���������
� �
D__inference_NN_model_layer_call_and_return_conditional_losses_414676j%&-.7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
D__inference_NN_model_layer_call_and_return_conditional_losses_414707j%&-.7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
)__inference_NN_model_layer_call_fn_414399a%&-.;�8
1�.
$�!

InputLayer���������
p 

 
� "�����������
)__inference_NN_model_layer_call_fn_414526a%&-.;�8
1�.
$�!

InputLayer���������
p

 
� "�����������
)__inference_NN_model_layer_call_fn_414624]%&-.7�4
-�*
 �
inputs���������
p 

 
� "�����������
)__inference_NN_model_layer_call_fn_414645]%&-.7�4
-�*
 �
inputs���������
p

 
� "�����������
G__inference_OutputLayer_layer_call_and_return_conditional_losses_414786\-./�,
%�"
 �
inputs���������#
� "%�"
�
0���������
� 
,__inference_OutputLayer_layer_call_fn_414776O-./�,
%�"
 �
inputs���������#
� "�����������
!__inference__wrapped_model_414305z%&-.3�0
)�&
$�!

InputLayer���������
� "9�6
4
OutputLayer%�"
OutputLayer����������
$__inference_signature_wrapper_414603�%&-.A�>
� 
7�4
2

InputLayer$�!

InputLayer���������"9�6
4
OutputLayer%�"
OutputLayer���������