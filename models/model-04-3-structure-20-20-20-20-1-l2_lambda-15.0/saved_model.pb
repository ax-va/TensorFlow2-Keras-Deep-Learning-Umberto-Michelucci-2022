��

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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.12v2.9.0-18-gd8ce9f9c3018��	
�
Adam/dense_79/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_79/bias/v
y
(Adam/dense_79/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_79/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_79/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_79/kernel/v
�
*Adam/dense_79/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_79/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_78/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_78/bias/v
y
(Adam/dense_78/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_78/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_78/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_78/kernel/v
�
*Adam/dense_78/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_78/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_77/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_77/bias/v
y
(Adam/dense_77/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_77/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_77/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_77/kernel/v
�
*Adam/dense_77/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_77/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_76/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_76/bias/v
y
(Adam/dense_76/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_76/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_76/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_76/kernel/v
�
*Adam/dense_76/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_76/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_75/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_75/bias/v
y
(Adam/dense_75/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_75/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_75/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_75/kernel/v
�
*Adam/dense_75/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_75/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_79/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_79/bias/m
y
(Adam/dense_79/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_79/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_79/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_79/kernel/m
�
*Adam/dense_79/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_79/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_78/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_78/bias/m
y
(Adam/dense_78/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_78/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_78/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_78/kernel/m
�
*Adam/dense_78/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_78/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_77/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_77/bias/m
y
(Adam/dense_77/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_77/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_77/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_77/kernel/m
�
*Adam/dense_77/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_77/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_76/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_76/bias/m
y
(Adam/dense_76/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_76/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_76/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_76/kernel/m
�
*Adam/dense_76/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_76/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_75/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_75/bias/m
y
(Adam/dense_75/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_75/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_75/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_75/kernel/m
�
*Adam/dense_75/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_75/kernel/m*
_output_shapes

:*
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
r
dense_79/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_79/bias
k
!dense_79/bias/Read/ReadVariableOpReadVariableOpdense_79/bias*
_output_shapes
:*
dtype0
z
dense_79/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_79/kernel
s
#dense_79/kernel/Read/ReadVariableOpReadVariableOpdense_79/kernel*
_output_shapes

:*
dtype0
r
dense_78/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_78/bias
k
!dense_78/bias/Read/ReadVariableOpReadVariableOpdense_78/bias*
_output_shapes
:*
dtype0
z
dense_78/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_78/kernel
s
#dense_78/kernel/Read/ReadVariableOpReadVariableOpdense_78/kernel*
_output_shapes

:*
dtype0
r
dense_77/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_77/bias
k
!dense_77/bias/Read/ReadVariableOpReadVariableOpdense_77/bias*
_output_shapes
:*
dtype0
z
dense_77/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_77/kernel
s
#dense_77/kernel/Read/ReadVariableOpReadVariableOpdense_77/kernel*
_output_shapes

:*
dtype0
r
dense_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_76/bias
k
!dense_76/bias/Read/ReadVariableOpReadVariableOpdense_76/bias*
_output_shapes
:*
dtype0
z
dense_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_76/kernel
s
#dense_76/kernel/Read/ReadVariableOpReadVariableOpdense_76/kernel*
_output_shapes

:*
dtype0
r
dense_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_75/bias
k
!dense_75/bias/Read/ReadVariableOpReadVariableOpdense_75/bias*
_output_shapes
:*
dtype0
z
dense_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_75/kernel
s
#dense_75/kernel/Read/ReadVariableOpReadVariableOpdense_75/kernel*
_output_shapes

:*
dtype0

NoOpNoOp
�F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�E
value�EB�E B�E
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
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
J
0
1
2
3
&4
'5
.6
/7
68
79*
J
0
1
2
3
&4
'5
.6
/7
68
79*

80
91
:2
;3* 
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Atrace_0
Btrace_1
Ctrace_2
Dtrace_3* 
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
* 
�
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_ratem�m�m�m�&m�'m�.m�/m�6m�7m�v�v�v�v�&v�'v�.v�/v�6v�7v�*

Nserving_default* 

0
1*

0
1*
	
80* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
_Y
VARIABLE_VALUEdense_75/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_75/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
90* 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
_Y
VARIABLE_VALUEdense_76/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_76/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
	
:0* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 
_Y
VARIABLE_VALUEdense_77/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_77/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
	
;0* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
_Y
VARIABLE_VALUEdense_78/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_78/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
_Y
VARIABLE_VALUEdense_79/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_79/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

rtrace_0* 

strace_0* 

ttrace_0* 

utrace_0* 
* 
.
0
1
2
3
4
5*

v0
w1*
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
	
80* 
* 
* 
* 
* 
* 
* 
	
90* 
* 
* 
* 
* 
* 
* 
	
:0* 
* 
* 
* 
* 
* 
* 
	
;0* 
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
x	variables
y	keras_api
	ztotal
	{count*
I
|	variables
}	keras_api
	~total
	count
�
_fn_kwargs*

z0
{1*

x	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

~0
1*

|	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�|
VARIABLE_VALUEAdam/dense_75/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_75/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_76/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_76/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_77/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_77/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_78/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_78/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_79/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_79/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_75/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_75/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_76/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_76/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_77/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_77/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_78/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_78/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_79/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_79/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_16Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_16dense_75/kerneldense_75/biasdense_76/kerneldense_76/biasdense_77/kerneldense_77/biasdense_78/kerneldense_78/biasdense_79/kerneldense_79/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3558872
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_75/kernel/Read/ReadVariableOp!dense_75/bias/Read/ReadVariableOp#dense_76/kernel/Read/ReadVariableOp!dense_76/bias/Read/ReadVariableOp#dense_77/kernel/Read/ReadVariableOp!dense_77/bias/Read/ReadVariableOp#dense_78/kernel/Read/ReadVariableOp!dense_78/bias/Read/ReadVariableOp#dense_79/kernel/Read/ReadVariableOp!dense_79/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_75/kernel/m/Read/ReadVariableOp(Adam/dense_75/bias/m/Read/ReadVariableOp*Adam/dense_76/kernel/m/Read/ReadVariableOp(Adam/dense_76/bias/m/Read/ReadVariableOp*Adam/dense_77/kernel/m/Read/ReadVariableOp(Adam/dense_77/bias/m/Read/ReadVariableOp*Adam/dense_78/kernel/m/Read/ReadVariableOp(Adam/dense_78/bias/m/Read/ReadVariableOp*Adam/dense_79/kernel/m/Read/ReadVariableOp(Adam/dense_79/bias/m/Read/ReadVariableOp*Adam/dense_75/kernel/v/Read/ReadVariableOp(Adam/dense_75/bias/v/Read/ReadVariableOp*Adam/dense_76/kernel/v/Read/ReadVariableOp(Adam/dense_76/bias/v/Read/ReadVariableOp*Adam/dense_77/kernel/v/Read/ReadVariableOp(Adam/dense_77/bias/v/Read/ReadVariableOp*Adam/dense_78/kernel/v/Read/ReadVariableOp(Adam/dense_78/bias/v/Read/ReadVariableOp*Adam/dense_79/kernel/v/Read/ReadVariableOp(Adam/dense_79/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_3559377
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_75/kerneldense_75/biasdense_76/kerneldense_76/biasdense_77/kerneldense_77/biasdense_78/kerneldense_78/biasdense_79/kerneldense_79/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_75/kernel/mAdam/dense_75/bias/mAdam/dense_76/kernel/mAdam/dense_76/bias/mAdam/dense_77/kernel/mAdam/dense_77/bias/mAdam/dense_78/kernel/mAdam/dense_78/bias/mAdam/dense_79/kernel/mAdam/dense_79/bias/mAdam/dense_75/kernel/vAdam/dense_75/bias/vAdam/dense_76/kernel/vAdam/dense_76/bias/vAdam/dense_77/kernel/vAdam/dense_77/bias/vAdam/dense_78/kernel/vAdam/dense_78/bias/vAdam/dense_79/kernel/vAdam/dense_79/bias/v*3
Tin,
*2(*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_3559504��
�
�
__inference_loss_fn_0_3559204L
:dense_75_kernel_regularizer_square_readvariableop_resource:
identity��1dense_75/kernel/Regularizer/Square/ReadVariableOp�
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_75_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_75/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_75/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp
�

�
4__inference_feed-forward-model_layer_call_fn_3558531
input_16
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_16
�	
�
E__inference_dense_79_layer_call_and_return_conditional_losses_3558477

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�<
�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558762
input_16"
dense_75_3558712:
dense_75_3558714:"
dense_76_3558717:
dense_76_3558719:"
dense_77_3558722:
dense_77_3558724:"
dense_78_3558727:
dense_78_3558729:"
dense_79_3558732:
dense_79_3558734:
identity�� dense_75/StatefulPartitionedCall�1dense_75/kernel/Regularizer/Square/ReadVariableOp� dense_76/StatefulPartitionedCall�1dense_76/kernel/Regularizer/Square/ReadVariableOp� dense_77/StatefulPartitionedCall�1dense_77/kernel/Regularizer/Square/ReadVariableOp� dense_78/StatefulPartitionedCall�1dense_78/kernel/Regularizer/Square/ReadVariableOp� dense_79/StatefulPartitionedCall�
 dense_75/StatefulPartitionedCallStatefulPartitionedCallinput_16dense_75_3558712dense_75_3558714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_3558392�
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_3558717dense_76_3558719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_3558415�
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_3558722dense_77_3558724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_77_layer_call_and_return_conditional_losses_3558438�
 dense_78/StatefulPartitionedCallStatefulPartitionedCall)dense_77/StatefulPartitionedCall:output:0dense_78_3558727dense_78_3558729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_78_layer_call_and_return_conditional_losses_3558461�
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_3558732dense_79_3558734*
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
GPU 2J 8� *N
fIRG
E__inference_dense_79_layer_call_and_return_conditional_losses_3558477�
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_75_3558712*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_76_3558717*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_77_3558722*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_78_3558727*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_79/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_75/StatefulPartitionedCall2^dense_75/kernel/Regularizer/Square/ReadVariableOp!^dense_76/StatefulPartitionedCall2^dense_76/kernel/Regularizer/Square/ReadVariableOp!^dense_77/StatefulPartitionedCall2^dense_77/kernel/Regularizer/Square/ReadVariableOp!^dense_78/StatefulPartitionedCall2^dense_78/kernel/Regularizer/Square/ReadVariableOp!^dense_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_16
�	
�
E__inference_dense_79_layer_call_and_return_conditional_losses_3559193

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_75_layer_call_and_return_conditional_losses_3558392

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_75/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_75/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
4__inference_feed-forward-model_layer_call_fn_3558946

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558661o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_77_layer_call_fn_3559131

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_77_layer_call_and_return_conditional_losses_3558438o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�<
�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558815
input_16"
dense_75_3558765:
dense_75_3558767:"
dense_76_3558770:
dense_76_3558772:"
dense_77_3558775:
dense_77_3558777:"
dense_78_3558780:
dense_78_3558782:"
dense_79_3558785:
dense_79_3558787:
identity�� dense_75/StatefulPartitionedCall�1dense_75/kernel/Regularizer/Square/ReadVariableOp� dense_76/StatefulPartitionedCall�1dense_76/kernel/Regularizer/Square/ReadVariableOp� dense_77/StatefulPartitionedCall�1dense_77/kernel/Regularizer/Square/ReadVariableOp� dense_78/StatefulPartitionedCall�1dense_78/kernel/Regularizer/Square/ReadVariableOp� dense_79/StatefulPartitionedCall�
 dense_75/StatefulPartitionedCallStatefulPartitionedCallinput_16dense_75_3558765dense_75_3558767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_3558392�
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_3558770dense_76_3558772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_3558415�
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_3558775dense_77_3558777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_77_layer_call_and_return_conditional_losses_3558438�
 dense_78/StatefulPartitionedCallStatefulPartitionedCall)dense_77/StatefulPartitionedCall:output:0dense_78_3558780dense_78_3558782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_78_layer_call_and_return_conditional_losses_3558461�
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_3558785dense_79_3558787*
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
GPU 2J 8� *N
fIRG
E__inference_dense_79_layer_call_and_return_conditional_losses_3558477�
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_75_3558765*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_76_3558770*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_77_3558775*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_78_3558780*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_79/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_75/StatefulPartitionedCall2^dense_75/kernel/Regularizer/Square/ReadVariableOp!^dense_76/StatefulPartitionedCall2^dense_76/kernel/Regularizer/Square/ReadVariableOp!^dense_77/StatefulPartitionedCall2^dense_77/kernel/Regularizer/Square/ReadVariableOp!^dense_78/StatefulPartitionedCall2^dense_78/kernel/Regularizer/Square/ReadVariableOp!^dense_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_16
�
�
E__inference_dense_77_layer_call_and_return_conditional_losses_3559148

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_77/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_77/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_76_layer_call_and_return_conditional_losses_3558415

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_76/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_75_layer_call_and_return_conditional_losses_3559096

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_75/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_75/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_3559215L
:dense_76_kernel_regularizer_square_readvariableop_resource:
identity��1dense_76/kernel/Regularizer/Square/ReadVariableOp�
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_76_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_76/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp
�

�
4__inference_feed-forward-model_layer_call_fn_3558921

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558508

inputs"
dense_75_3558393:
dense_75_3558395:"
dense_76_3558416:
dense_76_3558418:"
dense_77_3558439:
dense_77_3558441:"
dense_78_3558462:
dense_78_3558464:"
dense_79_3558478:
dense_79_3558480:
identity�� dense_75/StatefulPartitionedCall�1dense_75/kernel/Regularizer/Square/ReadVariableOp� dense_76/StatefulPartitionedCall�1dense_76/kernel/Regularizer/Square/ReadVariableOp� dense_77/StatefulPartitionedCall�1dense_77/kernel/Regularizer/Square/ReadVariableOp� dense_78/StatefulPartitionedCall�1dense_78/kernel/Regularizer/Square/ReadVariableOp� dense_79/StatefulPartitionedCall�
 dense_75/StatefulPartitionedCallStatefulPartitionedCallinputsdense_75_3558393dense_75_3558395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_3558392�
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_3558416dense_76_3558418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_3558415�
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_3558439dense_77_3558441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_77_layer_call_and_return_conditional_losses_3558438�
 dense_78/StatefulPartitionedCallStatefulPartitionedCall)dense_77/StatefulPartitionedCall:output:0dense_78_3558462dense_78_3558464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_78_layer_call_and_return_conditional_losses_3558461�
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_3558478dense_79_3558480*
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
GPU 2J 8� *N
fIRG
E__inference_dense_79_layer_call_and_return_conditional_losses_3558477�
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_75_3558393*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_76_3558416*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_77_3558439*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_78_3558462*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_79/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_75/StatefulPartitionedCall2^dense_75/kernel/Regularizer/Square/ReadVariableOp!^dense_76/StatefulPartitionedCall2^dense_76/kernel/Regularizer/Square/ReadVariableOp!^dense_77/StatefulPartitionedCall2^dense_77/kernel/Regularizer/Square/ReadVariableOp!^dense_78/StatefulPartitionedCall2^dense_78/kernel/Regularizer/Square/ReadVariableOp!^dense_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_76_layer_call_fn_3559105

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_3558415o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_3559226L
:dense_77_kernel_regularizer_square_readvariableop_resource:
identity��1dense_77/kernel/Regularizer/Square/ReadVariableOp�
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_77_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_77/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_77/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp
�P
�
 __inference__traced_save_3559377
file_prefix.
*savev2_dense_75_kernel_read_readvariableop,
(savev2_dense_75_bias_read_readvariableop.
*savev2_dense_76_kernel_read_readvariableop,
(savev2_dense_76_bias_read_readvariableop.
*savev2_dense_77_kernel_read_readvariableop,
(savev2_dense_77_bias_read_readvariableop.
*savev2_dense_78_kernel_read_readvariableop,
(savev2_dense_78_bias_read_readvariableop.
*savev2_dense_79_kernel_read_readvariableop,
(savev2_dense_79_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_75_kernel_m_read_readvariableop3
/savev2_adam_dense_75_bias_m_read_readvariableop5
1savev2_adam_dense_76_kernel_m_read_readvariableop3
/savev2_adam_dense_76_bias_m_read_readvariableop5
1savev2_adam_dense_77_kernel_m_read_readvariableop3
/savev2_adam_dense_77_bias_m_read_readvariableop5
1savev2_adam_dense_78_kernel_m_read_readvariableop3
/savev2_adam_dense_78_bias_m_read_readvariableop5
1savev2_adam_dense_79_kernel_m_read_readvariableop3
/savev2_adam_dense_79_bias_m_read_readvariableop5
1savev2_adam_dense_75_kernel_v_read_readvariableop3
/savev2_adam_dense_75_bias_v_read_readvariableop5
1savev2_adam_dense_76_kernel_v_read_readvariableop3
/savev2_adam_dense_76_bias_v_read_readvariableop5
1savev2_adam_dense_77_kernel_v_read_readvariableop3
/savev2_adam_dense_77_bias_v_read_readvariableop5
1savev2_adam_dense_78_kernel_v_read_readvariableop3
/savev2_adam_dense_78_bias_v_read_readvariableop5
1savev2_adam_dense_79_kernel_v_read_readvariableop3
/savev2_adam_dense_79_bias_v_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_75_kernel_read_readvariableop(savev2_dense_75_bias_read_readvariableop*savev2_dense_76_kernel_read_readvariableop(savev2_dense_76_bias_read_readvariableop*savev2_dense_77_kernel_read_readvariableop(savev2_dense_77_bias_read_readvariableop*savev2_dense_78_kernel_read_readvariableop(savev2_dense_78_bias_read_readvariableop*savev2_dense_79_kernel_read_readvariableop(savev2_dense_79_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_75_kernel_m_read_readvariableop/savev2_adam_dense_75_bias_m_read_readvariableop1savev2_adam_dense_76_kernel_m_read_readvariableop/savev2_adam_dense_76_bias_m_read_readvariableop1savev2_adam_dense_77_kernel_m_read_readvariableop/savev2_adam_dense_77_bias_m_read_readvariableop1savev2_adam_dense_78_kernel_m_read_readvariableop/savev2_adam_dense_78_bias_m_read_readvariableop1savev2_adam_dense_79_kernel_m_read_readvariableop/savev2_adam_dense_79_bias_m_read_readvariableop1savev2_adam_dense_75_kernel_v_read_readvariableop/savev2_adam_dense_75_bias_v_read_readvariableop1savev2_adam_dense_76_kernel_v_read_readvariableop/savev2_adam_dense_76_bias_v_read_readvariableop1savev2_adam_dense_77_kernel_v_read_readvariableop/savev2_adam_dense_77_bias_v_read_readvariableop1savev2_adam_dense_78_kernel_v_read_readvariableop/savev2_adam_dense_78_bias_v_read_readvariableop1savev2_adam_dense_79_kernel_v_read_readvariableop/savev2_adam_dense_79_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::: : : : : : : : : ::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::(

_output_shapes
: 
ț
�
#__inference__traced_restore_3559504
file_prefix2
 assignvariableop_dense_75_kernel:.
 assignvariableop_1_dense_75_bias:4
"assignvariableop_2_dense_76_kernel:.
 assignvariableop_3_dense_76_bias:4
"assignvariableop_4_dense_77_kernel:.
 assignvariableop_5_dense_77_bias:4
"assignvariableop_6_dense_78_kernel:.
 assignvariableop_7_dense_78_bias:4
"assignvariableop_8_dense_79_kernel:.
 assignvariableop_9_dense_79_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: <
*assignvariableop_19_adam_dense_75_kernel_m:6
(assignvariableop_20_adam_dense_75_bias_m:<
*assignvariableop_21_adam_dense_76_kernel_m:6
(assignvariableop_22_adam_dense_76_bias_m:<
*assignvariableop_23_adam_dense_77_kernel_m:6
(assignvariableop_24_adam_dense_77_bias_m:<
*assignvariableop_25_adam_dense_78_kernel_m:6
(assignvariableop_26_adam_dense_78_bias_m:<
*assignvariableop_27_adam_dense_79_kernel_m:6
(assignvariableop_28_adam_dense_79_bias_m:<
*assignvariableop_29_adam_dense_75_kernel_v:6
(assignvariableop_30_adam_dense_75_bias_v:<
*assignvariableop_31_adam_dense_76_kernel_v:6
(assignvariableop_32_adam_dense_76_bias_v:<
*assignvariableop_33_adam_dense_77_kernel_v:6
(assignvariableop_34_adam_dense_77_bias_v:<
*assignvariableop_35_adam_dense_78_kernel_v:6
(assignvariableop_36_adam_dense_78_bias_v:<
*assignvariableop_37_adam_dense_79_kernel_v:6
(assignvariableop_38_adam_dense_79_bias_v:
identity_40��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_75_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_75_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_76_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_76_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_77_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_77_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_78_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_78_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_79_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_79_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_75_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_75_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_76_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_76_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_77_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_77_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_78_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_78_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_79_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_79_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_75_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_75_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_76_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_76_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_77_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_77_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_78_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_78_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_79_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_79_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
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
�L
�	
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559070

inputs9
'dense_75_matmul_readvariableop_resource:6
(dense_75_biasadd_readvariableop_resource:9
'dense_76_matmul_readvariableop_resource:6
(dense_76_biasadd_readvariableop_resource:9
'dense_77_matmul_readvariableop_resource:6
(dense_77_biasadd_readvariableop_resource:9
'dense_78_matmul_readvariableop_resource:6
(dense_78_biasadd_readvariableop_resource:9
'dense_79_matmul_readvariableop_resource:6
(dense_79_biasadd_readvariableop_resource:
identity��dense_75/BiasAdd/ReadVariableOp�dense_75/MatMul/ReadVariableOp�1dense_75/kernel/Regularizer/Square/ReadVariableOp�dense_76/BiasAdd/ReadVariableOp�dense_76/MatMul/ReadVariableOp�1dense_76/kernel/Regularizer/Square/ReadVariableOp�dense_77/BiasAdd/ReadVariableOp�dense_77/MatMul/ReadVariableOp�1dense_77/kernel/Regularizer/Square/ReadVariableOp�dense_78/BiasAdd/ReadVariableOp�dense_78/MatMul/ReadVariableOp�1dense_78/kernel/Regularizer/Square/ReadVariableOp�dense_79/BiasAdd/ReadVariableOp�dense_79/MatMul/ReadVariableOp�
dense_75/MatMul/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_75/MatMulMatMulinputs&dense_75/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_75/BiasAdd/ReadVariableOpReadVariableOp(dense_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_75/BiasAddBiasAdddense_75/MatMul:product:0'dense_75/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_75/ReluReludense_75/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_76/MatMul/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_76/MatMulMatMuldense_75/Relu:activations:0&dense_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_76/BiasAddBiasAdddense_76/MatMul:product:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_76/ReluReludense_76/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_77/MatMul/ReadVariableOpReadVariableOp'dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_77/MatMulMatMuldense_76/Relu:activations:0&dense_77/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_77/BiasAdd/ReadVariableOpReadVariableOp(dense_77_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_77/BiasAddBiasAdddense_77/MatMul:product:0'dense_77/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_77/ReluReludense_77/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_78/MatMul/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_78/MatMulMatMuldense_77/Relu:activations:0&dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_78/BiasAdd/ReadVariableOpReadVariableOp(dense_78_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_78/BiasAddBiasAdddense_78/MatMul:product:0'dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_78/ReluReludense_78/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_79/MatMul/ReadVariableOpReadVariableOp'dense_79_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_79/MatMulMatMuldense_78/Relu:activations:0&dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_79/BiasAdd/ReadVariableOpReadVariableOp(dense_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_79/BiasAddBiasAdddense_79/MatMul:product:0'dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_79/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_75/BiasAdd/ReadVariableOp^dense_75/MatMul/ReadVariableOp2^dense_75/kernel/Regularizer/Square/ReadVariableOp ^dense_76/BiasAdd/ReadVariableOp^dense_76/MatMul/ReadVariableOp2^dense_76/kernel/Regularizer/Square/ReadVariableOp ^dense_77/BiasAdd/ReadVariableOp^dense_77/MatMul/ReadVariableOp2^dense_77/kernel/Regularizer/Square/ReadVariableOp ^dense_78/BiasAdd/ReadVariableOp^dense_78/MatMul/ReadVariableOp2^dense_78/kernel/Regularizer/Square/ReadVariableOp ^dense_79/BiasAdd/ReadVariableOp^dense_79/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2B
dense_75/BiasAdd/ReadVariableOpdense_75/BiasAdd/ReadVariableOp2@
dense_75/MatMul/ReadVariableOpdense_75/MatMul/ReadVariableOp2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2@
dense_76/MatMul/ReadVariableOpdense_76/MatMul/ReadVariableOp2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp2B
dense_77/BiasAdd/ReadVariableOpdense_77/BiasAdd/ReadVariableOp2@
dense_77/MatMul/ReadVariableOpdense_77/MatMul/ReadVariableOp2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp2B
dense_78/BiasAdd/ReadVariableOpdense_78/BiasAdd/ReadVariableOp2@
dense_78/MatMul/ReadVariableOpdense_78/MatMul/ReadVariableOp2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp2B
dense_79/BiasAdd/ReadVariableOpdense_79/BiasAdd/ReadVariableOp2@
dense_79/MatMul/ReadVariableOpdense_79/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
4__inference_feed-forward-model_layer_call_fn_3558709
input_16
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558661o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_16
�
�
__inference_loss_fn_3_3559237L
:dense_78_kernel_regularizer_square_readvariableop_resource:
identity��1dense_78/kernel/Regularizer/Square/ReadVariableOp�
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_78_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_78/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_78/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp
�
�
E__inference_dense_76_layer_call_and_return_conditional_losses_3559122

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_76/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_76/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558661

inputs"
dense_75_3558611:
dense_75_3558613:"
dense_76_3558616:
dense_76_3558618:"
dense_77_3558621:
dense_77_3558623:"
dense_78_3558626:
dense_78_3558628:"
dense_79_3558631:
dense_79_3558633:
identity�� dense_75/StatefulPartitionedCall�1dense_75/kernel/Regularizer/Square/ReadVariableOp� dense_76/StatefulPartitionedCall�1dense_76/kernel/Regularizer/Square/ReadVariableOp� dense_77/StatefulPartitionedCall�1dense_77/kernel/Regularizer/Square/ReadVariableOp� dense_78/StatefulPartitionedCall�1dense_78/kernel/Regularizer/Square/ReadVariableOp� dense_79/StatefulPartitionedCall�
 dense_75/StatefulPartitionedCallStatefulPartitionedCallinputsdense_75_3558611dense_75_3558613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_3558392�
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_3558616dense_76_3558618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_76_layer_call_and_return_conditional_losses_3558415�
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_3558621dense_77_3558623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_77_layer_call_and_return_conditional_losses_3558438�
 dense_78/StatefulPartitionedCallStatefulPartitionedCall)dense_77/StatefulPartitionedCall:output:0dense_78_3558626dense_78_3558628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_78_layer_call_and_return_conditional_losses_3558461�
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_3558631dense_79_3558633*
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
GPU 2J 8� *N
fIRG
E__inference_dense_79_layer_call_and_return_conditional_losses_3558477�
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_75_3558611*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_76_3558616*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_77_3558621*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_78_3558626*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_79/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_75/StatefulPartitionedCall2^dense_75/kernel/Regularizer/Square/ReadVariableOp!^dense_76/StatefulPartitionedCall2^dense_76/kernel/Regularizer/Square/ReadVariableOp!^dense_77/StatefulPartitionedCall2^dense_77/kernel/Regularizer/Square/ReadVariableOp!^dense_78/StatefulPartitionedCall2^dense_78/kernel/Regularizer/Square/ReadVariableOp!^dense_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_78_layer_call_fn_3559157

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_78_layer_call_and_return_conditional_losses_3558461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_77_layer_call_and_return_conditional_losses_3558438

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_77/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_77/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�

"__inference__wrapped_model_3558368
input_16L
:feed_forward_model_dense_75_matmul_readvariableop_resource:I
;feed_forward_model_dense_75_biasadd_readvariableop_resource:L
:feed_forward_model_dense_76_matmul_readvariableop_resource:I
;feed_forward_model_dense_76_biasadd_readvariableop_resource:L
:feed_forward_model_dense_77_matmul_readvariableop_resource:I
;feed_forward_model_dense_77_biasadd_readvariableop_resource:L
:feed_forward_model_dense_78_matmul_readvariableop_resource:I
;feed_forward_model_dense_78_biasadd_readvariableop_resource:L
:feed_forward_model_dense_79_matmul_readvariableop_resource:I
;feed_forward_model_dense_79_biasadd_readvariableop_resource:
identity��2feed-forward-model/dense_75/BiasAdd/ReadVariableOp�1feed-forward-model/dense_75/MatMul/ReadVariableOp�2feed-forward-model/dense_76/BiasAdd/ReadVariableOp�1feed-forward-model/dense_76/MatMul/ReadVariableOp�2feed-forward-model/dense_77/BiasAdd/ReadVariableOp�1feed-forward-model/dense_77/MatMul/ReadVariableOp�2feed-forward-model/dense_78/BiasAdd/ReadVariableOp�1feed-forward-model/dense_78/MatMul/ReadVariableOp�2feed-forward-model/dense_79/BiasAdd/ReadVariableOp�1feed-forward-model/dense_79/MatMul/ReadVariableOp�
1feed-forward-model/dense_75/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_75_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_75/MatMulMatMulinput_169feed-forward-model/dense_75/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_75/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_75/BiasAddBiasAdd,feed-forward-model/dense_75/MatMul:product:0:feed-forward-model/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_75/ReluRelu,feed-forward-model/dense_75/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_76/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_76/MatMulMatMul.feed-forward-model/dense_75/Relu:activations:09feed-forward-model/dense_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_76/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_76/BiasAddBiasAdd,feed-forward-model/dense_76/MatMul:product:0:feed-forward-model/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_76/ReluRelu,feed-forward-model/dense_76/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_77/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_77/MatMulMatMul.feed-forward-model/dense_76/Relu:activations:09feed-forward-model/dense_77/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_77/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_77_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_77/BiasAddBiasAdd,feed-forward-model/dense_77/MatMul:product:0:feed-forward-model/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_77/ReluRelu,feed-forward-model/dense_77/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_78/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_78_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_78/MatMulMatMul.feed-forward-model/dense_77/Relu:activations:09feed-forward-model/dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_78/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_78_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_78/BiasAddBiasAdd,feed-forward-model/dense_78/MatMul:product:0:feed-forward-model/dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_78/ReluRelu,feed-forward-model/dense_78/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_79/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_79_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_79/MatMulMatMul.feed-forward-model/dense_78/Relu:activations:09feed-forward-model/dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_79/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_79/BiasAddBiasAdd,feed-forward-model/dense_79/MatMul:product:0:feed-forward-model/dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,feed-forward-model/dense_79/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^feed-forward-model/dense_75/BiasAdd/ReadVariableOp2^feed-forward-model/dense_75/MatMul/ReadVariableOp3^feed-forward-model/dense_76/BiasAdd/ReadVariableOp2^feed-forward-model/dense_76/MatMul/ReadVariableOp3^feed-forward-model/dense_77/BiasAdd/ReadVariableOp2^feed-forward-model/dense_77/MatMul/ReadVariableOp3^feed-forward-model/dense_78/BiasAdd/ReadVariableOp2^feed-forward-model/dense_78/MatMul/ReadVariableOp3^feed-forward-model/dense_79/BiasAdd/ReadVariableOp2^feed-forward-model/dense_79/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2h
2feed-forward-model/dense_75/BiasAdd/ReadVariableOp2feed-forward-model/dense_75/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_75/MatMul/ReadVariableOp1feed-forward-model/dense_75/MatMul/ReadVariableOp2h
2feed-forward-model/dense_76/BiasAdd/ReadVariableOp2feed-forward-model/dense_76/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_76/MatMul/ReadVariableOp1feed-forward-model/dense_76/MatMul/ReadVariableOp2h
2feed-forward-model/dense_77/BiasAdd/ReadVariableOp2feed-forward-model/dense_77/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_77/MatMul/ReadVariableOp1feed-forward-model/dense_77/MatMul/ReadVariableOp2h
2feed-forward-model/dense_78/BiasAdd/ReadVariableOp2feed-forward-model/dense_78/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_78/MatMul/ReadVariableOp1feed-forward-model/dense_78/MatMul/ReadVariableOp2h
2feed-forward-model/dense_79/BiasAdd/ReadVariableOp2feed-forward-model/dense_79/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_79/MatMul/ReadVariableOp1feed-forward-model/dense_79/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_16
�	
�
%__inference_signature_wrapper_3558872
input_16
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_3558368o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_16
�L
�	
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559008

inputs9
'dense_75_matmul_readvariableop_resource:6
(dense_75_biasadd_readvariableop_resource:9
'dense_76_matmul_readvariableop_resource:6
(dense_76_biasadd_readvariableop_resource:9
'dense_77_matmul_readvariableop_resource:6
(dense_77_biasadd_readvariableop_resource:9
'dense_78_matmul_readvariableop_resource:6
(dense_78_biasadd_readvariableop_resource:9
'dense_79_matmul_readvariableop_resource:6
(dense_79_biasadd_readvariableop_resource:
identity��dense_75/BiasAdd/ReadVariableOp�dense_75/MatMul/ReadVariableOp�1dense_75/kernel/Regularizer/Square/ReadVariableOp�dense_76/BiasAdd/ReadVariableOp�dense_76/MatMul/ReadVariableOp�1dense_76/kernel/Regularizer/Square/ReadVariableOp�dense_77/BiasAdd/ReadVariableOp�dense_77/MatMul/ReadVariableOp�1dense_77/kernel/Regularizer/Square/ReadVariableOp�dense_78/BiasAdd/ReadVariableOp�dense_78/MatMul/ReadVariableOp�1dense_78/kernel/Regularizer/Square/ReadVariableOp�dense_79/BiasAdd/ReadVariableOp�dense_79/MatMul/ReadVariableOp�
dense_75/MatMul/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_75/MatMulMatMulinputs&dense_75/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_75/BiasAdd/ReadVariableOpReadVariableOp(dense_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_75/BiasAddBiasAdddense_75/MatMul:product:0'dense_75/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_75/ReluReludense_75/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_76/MatMul/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_76/MatMulMatMuldense_75/Relu:activations:0&dense_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_76/BiasAddBiasAdddense_76/MatMul:product:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_76/ReluReludense_76/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_77/MatMul/ReadVariableOpReadVariableOp'dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_77/MatMulMatMuldense_76/Relu:activations:0&dense_77/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_77/BiasAdd/ReadVariableOpReadVariableOp(dense_77_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_77/BiasAddBiasAdddense_77/MatMul:product:0'dense_77/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_77/ReluReludense_77/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_78/MatMul/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_78/MatMulMatMuldense_77/Relu:activations:0&dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_78/BiasAdd/ReadVariableOpReadVariableOp(dense_78_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_78/BiasAddBiasAdddense_78/MatMul:product:0'dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_78/ReluReludense_78/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_79/MatMul/ReadVariableOpReadVariableOp'dense_79_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_79/MatMulMatMuldense_78/Relu:activations:0&dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_79/BiasAdd/ReadVariableOpReadVariableOp(dense_79_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_79/BiasAddBiasAdddense_79/MatMul:product:0'dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_75/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_75/kernel/Regularizer/SquareSquare9dense_75/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_75/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_75/kernel/Regularizer/SumSum&dense_75/kernel/Regularizer/Square:y:0*dense_75/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_75/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_75/kernel/Regularizer/mulMul*dense_75/kernel/Regularizer/mul/x:output:0(dense_75/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_76/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_76/kernel/Regularizer/SquareSquare9dense_76/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_76/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_76/kernel/Regularizer/SumSum&dense_76/kernel/Regularizer/Square:y:0*dense_76/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_76/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_76/kernel/Regularizer/mulMul*dense_76/kernel/Regularizer/mul/x:output:0(dense_76/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_77/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_77/kernel/Regularizer/SquareSquare9dense_77/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_77/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_77/kernel/Regularizer/SumSum&dense_77/kernel/Regularizer/Square:y:0*dense_77/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_77/kernel/Regularizer/mulMul*dense_77/kernel/Regularizer/mul/x:output:0(dense_77/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_79/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_75/BiasAdd/ReadVariableOp^dense_75/MatMul/ReadVariableOp2^dense_75/kernel/Regularizer/Square/ReadVariableOp ^dense_76/BiasAdd/ReadVariableOp^dense_76/MatMul/ReadVariableOp2^dense_76/kernel/Regularizer/Square/ReadVariableOp ^dense_77/BiasAdd/ReadVariableOp^dense_77/MatMul/ReadVariableOp2^dense_77/kernel/Regularizer/Square/ReadVariableOp ^dense_78/BiasAdd/ReadVariableOp^dense_78/MatMul/ReadVariableOp2^dense_78/kernel/Regularizer/Square/ReadVariableOp ^dense_79/BiasAdd/ReadVariableOp^dense_79/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2B
dense_75/BiasAdd/ReadVariableOpdense_75/BiasAdd/ReadVariableOp2@
dense_75/MatMul/ReadVariableOpdense_75/MatMul/ReadVariableOp2f
1dense_75/kernel/Regularizer/Square/ReadVariableOp1dense_75/kernel/Regularizer/Square/ReadVariableOp2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2@
dense_76/MatMul/ReadVariableOpdense_76/MatMul/ReadVariableOp2f
1dense_76/kernel/Regularizer/Square/ReadVariableOp1dense_76/kernel/Regularizer/Square/ReadVariableOp2B
dense_77/BiasAdd/ReadVariableOpdense_77/BiasAdd/ReadVariableOp2@
dense_77/MatMul/ReadVariableOpdense_77/MatMul/ReadVariableOp2f
1dense_77/kernel/Regularizer/Square/ReadVariableOp1dense_77/kernel/Regularizer/Square/ReadVariableOp2B
dense_78/BiasAdd/ReadVariableOpdense_78/BiasAdd/ReadVariableOp2@
dense_78/MatMul/ReadVariableOpdense_78/MatMul/ReadVariableOp2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp2B
dense_79/BiasAdd/ReadVariableOpdense_79/BiasAdd/ReadVariableOp2@
dense_79/MatMul/ReadVariableOpdense_79/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_75_layer_call_fn_3559079

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_75_layer_call_and_return_conditional_losses_3558392o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_78_layer_call_and_return_conditional_losses_3559174

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_78/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_78/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_79_layer_call_fn_3559183

inputs
unknown:
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
GPU 2J 8� *N
fIRG
E__inference_dense_79_layer_call_and_return_conditional_losses_3558477o
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
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_78_layer_call_and_return_conditional_losses_3558461

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_78/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_78/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_78/kernel/Regularizer/SquareSquare9dense_78/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_78/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_78/kernel/Regularizer/SumSum&dense_78/kernel/Regularizer/Square:y:0*dense_78/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_78/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  pA�
dense_78/kernel/Regularizer/mulMul*dense_78/kernel/Regularizer/mul/x:output:0(dense_78/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_78/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_78/kernel/Regularizer/Square/ReadVariableOp1dense_78/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_161
serving_default_input_16:0���������<
dense_790
StatefulPartitionedCall:0���������tensorflow/serving/predict:̤
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
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
f
0
1
2
3
&4
'5
.6
/7
68
79"
trackable_list_wrapper
f
0
1
2
3
&4
'5
.6
/7
68
79"
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Atrace_0
Btrace_1
Ctrace_2
Dtrace_32�
4__inference_feed-forward-model_layer_call_fn_3558531
4__inference_feed-forward-model_layer_call_fn_3558921
4__inference_feed-forward-model_layer_call_fn_3558946
4__inference_feed-forward-model_layer_call_fn_3558709�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zAtrace_0zBtrace_1zCtrace_2zDtrace_3
�
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559008
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559070
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558762
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558815�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
�B�
"__inference__wrapped_model_3558368input_16"�
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
�
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_ratem�m�m�m�&m�'m�.m�/m�6m�7m�v�v�v�v�&v�'v�.v�/v�6v�7v�"
	optimizer
,
Nserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
80"
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ttrace_02�
*__inference_dense_75_layer_call_fn_3559079�
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
 zTtrace_0
�
Utrace_02�
E__inference_dense_75_layer_call_and_return_conditional_losses_3559096�
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
!:2dense_75/kernel
:2dense_75/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
90"
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
[trace_02�
*__inference_dense_76_layer_call_fn_3559105�
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
 z[trace_0
�
\trace_02�
E__inference_dense_76_layer_call_and_return_conditional_losses_3559122�
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
!:2dense_76/kernel
:2dense_76/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
btrace_02�
*__inference_dense_77_layer_call_fn_3559131�
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
 zbtrace_0
�
ctrace_02�
E__inference_dense_77_layer_call_and_return_conditional_losses_3559148�
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
 zctrace_0
!:2dense_77/kernel
:2dense_77/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
'
;0"
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
itrace_02�
*__inference_dense_78_layer_call_fn_3559157�
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
 zitrace_0
�
jtrace_02�
E__inference_dense_78_layer_call_and_return_conditional_losses_3559174�
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
 zjtrace_0
!:2dense_78/kernel
:2dense_78/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
*__inference_dense_79_layer_call_fn_3559183�
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
 zptrace_0
�
qtrace_02�
E__inference_dense_79_layer_call_and_return_conditional_losses_3559193�
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
 zqtrace_0
!:2dense_79/kernel
:2dense_79/bias
�
rtrace_02�
__inference_loss_fn_0_3559204�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zrtrace_0
�
strace_02�
__inference_loss_fn_1_3559215�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zstrace_0
�
ttrace_02�
__inference_loss_fn_2_3559226�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zttrace_0
�
utrace_02�
__inference_loss_fn_3_3559237�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zutrace_0
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_feed-forward-model_layer_call_fn_3558531input_16"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
4__inference_feed-forward-model_layer_call_fn_3558921inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
4__inference_feed-forward-model_layer_call_fn_3558946inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
4__inference_feed-forward-model_layer_call_fn_3558709input_16"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559008inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559070inputs"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558762input_16"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558815input_16"�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
%__inference_signature_wrapper_3558872input_16"�
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
'
80"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_75_layer_call_fn_3559079inputs"�
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
E__inference_dense_75_layer_call_and_return_conditional_losses_3559096inputs"�
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
'
90"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_76_layer_call_fn_3559105inputs"�
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
E__inference_dense_76_layer_call_and_return_conditional_losses_3559122inputs"�
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
'
:0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_77_layer_call_fn_3559131inputs"�
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
E__inference_dense_77_layer_call_and_return_conditional_losses_3559148inputs"�
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
'
;0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_78_layer_call_fn_3559157inputs"�
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
E__inference_dense_78_layer_call_and_return_conditional_losses_3559174inputs"�
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
*__inference_dense_79_layer_call_fn_3559183inputs"�
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
E__inference_dense_79_layer_call_and_return_conditional_losses_3559193inputs"�
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
__inference_loss_fn_0_3559204"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_3559215"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_3559226"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_3559237"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
N
x	variables
y	keras_api
	ztotal
	{count"
_tf_keras_metric
_
|	variables
}	keras_api
	~total
	count
�
_fn_kwargs"
_tf_keras_metric
.
z0
{1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
&:$2Adam/dense_75/kernel/m
 :2Adam/dense_75/bias/m
&:$2Adam/dense_76/kernel/m
 :2Adam/dense_76/bias/m
&:$2Adam/dense_77/kernel/m
 :2Adam/dense_77/bias/m
&:$2Adam/dense_78/kernel/m
 :2Adam/dense_78/bias/m
&:$2Adam/dense_79/kernel/m
 :2Adam/dense_79/bias/m
&:$2Adam/dense_75/kernel/v
 :2Adam/dense_75/bias/v
&:$2Adam/dense_76/kernel/v
 :2Adam/dense_76/bias/v
&:$2Adam/dense_77/kernel/v
 :2Adam/dense_77/bias/v
&:$2Adam/dense_78/kernel/v
 :2Adam/dense_78/bias/v
&:$2Adam/dense_79/kernel/v
 :2Adam/dense_79/bias/v�
"__inference__wrapped_model_3558368t
&'./671�.
'�$
"�
input_16���������
� "3�0
.
dense_79"�
dense_79����������
E__inference_dense_75_layer_call_and_return_conditional_losses_3559096\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_75_layer_call_fn_3559079O/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_76_layer_call_and_return_conditional_losses_3559122\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_76_layer_call_fn_3559105O/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_77_layer_call_and_return_conditional_losses_3559148\&'/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_77_layer_call_fn_3559131O&'/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_78_layer_call_and_return_conditional_losses_3559174\.//�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_78_layer_call_fn_3559157O.//�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_79_layer_call_and_return_conditional_losses_3559193\67/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_79_layer_call_fn_3559183O67/�,
%�"
 �
inputs���������
� "�����������
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558762n
&'./679�6
/�,
"�
input_16���������
p 

 
� "%�"
�
0���������
� �
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3558815n
&'./679�6
/�,
"�
input_16���������
p

 
� "%�"
�
0���������
� �
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559008l
&'./677�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3559070l
&'./677�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
4__inference_feed-forward-model_layer_call_fn_3558531a
&'./679�6
/�,
"�
input_16���������
p 

 
� "�����������
4__inference_feed-forward-model_layer_call_fn_3558709a
&'./679�6
/�,
"�
input_16���������
p

 
� "�����������
4__inference_feed-forward-model_layer_call_fn_3558921_
&'./677�4
-�*
 �
inputs���������
p 

 
� "�����������
4__inference_feed-forward-model_layer_call_fn_3558946_
&'./677�4
-�*
 �
inputs���������
p

 
� "����������<
__inference_loss_fn_0_3559204�

� 
� "� <
__inference_loss_fn_1_3559215�

� 
� "� <
__inference_loss_fn_2_3559226&�

� 
� "� <
__inference_loss_fn_3_3559237.�

� 
� "� �
%__inference_signature_wrapper_3558872�
&'./67=�:
� 
3�0
.
input_16"�
input_16���������"3�0
.
dense_79"�
dense_79���������