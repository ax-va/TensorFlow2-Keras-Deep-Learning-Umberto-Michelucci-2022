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
Adam/dense_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_84/bias/v
y
(Adam/dense_84/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_84/kernel/v
�
*Adam/dense_84/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_83/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/v
y
(Adam/dense_83/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_83/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_83/kernel/v
�
*Adam/dense_83/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_82/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/v
y
(Adam/dense_82/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_82/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/v
�
*Adam/dense_82/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_81/bias/v
y
(Adam/dense_81/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_81/kernel/v
�
*Adam/dense_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_80/bias/v
y
(Adam/dense_80/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_80/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_80/kernel/v
�
*Adam/dense_80/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_80/kernel/v*
_output_shapes

:*
dtype0
�
Adam/dense_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_84/bias/m
y
(Adam/dense_84/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_84/kernel/m
�
*Adam/dense_84/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_84/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_83/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_83/bias/m
y
(Adam/dense_83/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_83/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_83/kernel/m
�
*Adam/dense_83/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_83/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_82/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_82/bias/m
y
(Adam/dense_82/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_82/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_82/kernel/m
�
*Adam/dense_82/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_82/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_81/bias/m
y
(Adam/dense_81/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_81/kernel/m
�
*Adam/dense_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_81/kernel/m*
_output_shapes

:*
dtype0
�
Adam/dense_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_80/bias/m
y
(Adam/dense_80/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_80/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_80/kernel/m
�
*Adam/dense_80/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_80/kernel/m*
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
dense_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_84/bias
k
!dense_84/bias/Read/ReadVariableOpReadVariableOpdense_84/bias*
_output_shapes
:*
dtype0
z
dense_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_84/kernel
s
#dense_84/kernel/Read/ReadVariableOpReadVariableOpdense_84/kernel*
_output_shapes

:*
dtype0
r
dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_83/bias
k
!dense_83/bias/Read/ReadVariableOpReadVariableOpdense_83/bias*
_output_shapes
:*
dtype0
z
dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_83/kernel
s
#dense_83/kernel/Read/ReadVariableOpReadVariableOpdense_83/kernel*
_output_shapes

:*
dtype0
r
dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_82/bias
k
!dense_82/bias/Read/ReadVariableOpReadVariableOpdense_82/bias*
_output_shapes
:*
dtype0
z
dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_82/kernel
s
#dense_82/kernel/Read/ReadVariableOpReadVariableOpdense_82/kernel*
_output_shapes

:*
dtype0
r
dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_81/bias
k
!dense_81/bias/Read/ReadVariableOpReadVariableOpdense_81/bias*
_output_shapes
:*
dtype0
z
dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_81/kernel
s
#dense_81/kernel/Read/ReadVariableOpReadVariableOpdense_81/kernel*
_output_shapes

:*
dtype0
r
dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_80/bias
k
!dense_80/bias/Read/ReadVariableOpReadVariableOpdense_80/bias*
_output_shapes
:*
dtype0
z
dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_80/kernel
s
#dense_80/kernel/Read/ReadVariableOpReadVariableOpdense_80/kernel*
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
VARIABLE_VALUEdense_80/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_80/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_81/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_81/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_82/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_82/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_83/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_83/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_84/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_84/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_80/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_80/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_81/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_81/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_82/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_83/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_84/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_84/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_80/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_80/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_81/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_81/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_82/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_82/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_83/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_83/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_84/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_84/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_17Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17dense_80/kerneldense_80/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/biasdense_84/kerneldense_84/bias*
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
%__inference_signature_wrapper_3781352
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_80/kernel/Read/ReadVariableOp!dense_80/bias/Read/ReadVariableOp#dense_81/kernel/Read/ReadVariableOp!dense_81/bias/Read/ReadVariableOp#dense_82/kernel/Read/ReadVariableOp!dense_82/bias/Read/ReadVariableOp#dense_83/kernel/Read/ReadVariableOp!dense_83/bias/Read/ReadVariableOp#dense_84/kernel/Read/ReadVariableOp!dense_84/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_80/kernel/m/Read/ReadVariableOp(Adam/dense_80/bias/m/Read/ReadVariableOp*Adam/dense_81/kernel/m/Read/ReadVariableOp(Adam/dense_81/bias/m/Read/ReadVariableOp*Adam/dense_82/kernel/m/Read/ReadVariableOp(Adam/dense_82/bias/m/Read/ReadVariableOp*Adam/dense_83/kernel/m/Read/ReadVariableOp(Adam/dense_83/bias/m/Read/ReadVariableOp*Adam/dense_84/kernel/m/Read/ReadVariableOp(Adam/dense_84/bias/m/Read/ReadVariableOp*Adam/dense_80/kernel/v/Read/ReadVariableOp(Adam/dense_80/bias/v/Read/ReadVariableOp*Adam/dense_81/kernel/v/Read/ReadVariableOp(Adam/dense_81/bias/v/Read/ReadVariableOp*Adam/dense_82/kernel/v/Read/ReadVariableOp(Adam/dense_82/bias/v/Read/ReadVariableOp*Adam/dense_83/kernel/v/Read/ReadVariableOp(Adam/dense_83/bias/v/Read/ReadVariableOp*Adam/dense_84/kernel/v/Read/ReadVariableOp(Adam/dense_84/bias/v/Read/ReadVariableOpConst*4
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
 __inference__traced_save_3781857
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_80/kerneldense_80/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/biasdense_84/kerneldense_84/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_80/kernel/mAdam/dense_80/bias/mAdam/dense_81/kernel/mAdam/dense_81/bias/mAdam/dense_82/kernel/mAdam/dense_82/bias/mAdam/dense_83/kernel/mAdam/dense_83/bias/mAdam/dense_84/kernel/mAdam/dense_84/bias/mAdam/dense_80/kernel/vAdam/dense_80/bias/vAdam/dense_81/kernel/vAdam/dense_81/bias/vAdam/dense_82/kernel/vAdam/dense_82/bias/vAdam/dense_83/kernel/vAdam/dense_83/bias/vAdam/dense_84/kernel/vAdam/dense_84/bias/v*3
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
#__inference__traced_restore_3781984��
�
�
E__inference_dense_83_layer_call_and_return_conditional_losses_3780941

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_83/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_83/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_81_layer_call_and_return_conditional_losses_3780895

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_3781706L
:dense_82_kernel_regularizer_square_readvariableop_resource:
identity��1dense_82/kernel/Regularizer/Square/ReadVariableOp�
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_82_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_82/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_82/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp
�;
�

"__inference__wrapped_model_3780848
input_17L
:feed_forward_model_dense_80_matmul_readvariableop_resource:I
;feed_forward_model_dense_80_biasadd_readvariableop_resource:L
:feed_forward_model_dense_81_matmul_readvariableop_resource:I
;feed_forward_model_dense_81_biasadd_readvariableop_resource:L
:feed_forward_model_dense_82_matmul_readvariableop_resource:I
;feed_forward_model_dense_82_biasadd_readvariableop_resource:L
:feed_forward_model_dense_83_matmul_readvariableop_resource:I
;feed_forward_model_dense_83_biasadd_readvariableop_resource:L
:feed_forward_model_dense_84_matmul_readvariableop_resource:I
;feed_forward_model_dense_84_biasadd_readvariableop_resource:
identity��2feed-forward-model/dense_80/BiasAdd/ReadVariableOp�1feed-forward-model/dense_80/MatMul/ReadVariableOp�2feed-forward-model/dense_81/BiasAdd/ReadVariableOp�1feed-forward-model/dense_81/MatMul/ReadVariableOp�2feed-forward-model/dense_82/BiasAdd/ReadVariableOp�1feed-forward-model/dense_82/MatMul/ReadVariableOp�2feed-forward-model/dense_83/BiasAdd/ReadVariableOp�1feed-forward-model/dense_83/MatMul/ReadVariableOp�2feed-forward-model/dense_84/BiasAdd/ReadVariableOp�1feed-forward-model/dense_84/MatMul/ReadVariableOp�
1feed-forward-model/dense_80/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_80/MatMulMatMulinput_179feed-forward-model/dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_80/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_80/BiasAddBiasAdd,feed-forward-model/dense_80/MatMul:product:0:feed-forward-model/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_80/ReluRelu,feed-forward-model/dense_80/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_81/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_81/MatMulMatMul.feed-forward-model/dense_80/Relu:activations:09feed-forward-model/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_81/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_81/BiasAddBiasAdd,feed-forward-model/dense_81/MatMul:product:0:feed-forward-model/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_81/ReluRelu,feed-forward-model/dense_81/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_82/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_82/MatMulMatMul.feed-forward-model/dense_81/Relu:activations:09feed-forward-model/dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_82/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_82/BiasAddBiasAdd,feed-forward-model/dense_82/MatMul:product:0:feed-forward-model/dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_82/ReluRelu,feed-forward-model/dense_82/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_83/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_83/MatMulMatMul.feed-forward-model/dense_82/Relu:activations:09feed-forward-model/dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_83/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_83/BiasAddBiasAdd,feed-forward-model/dense_83/MatMul:product:0:feed-forward-model/dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 feed-forward-model/dense_83/ReluRelu,feed-forward-model/dense_83/BiasAdd:output:0*
T0*'
_output_shapes
:����������
1feed-forward-model/dense_84/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_84_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"feed-forward-model/dense_84/MatMulMatMul.feed-forward-model/dense_83/Relu:activations:09feed-forward-model/dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2feed-forward-model/dense_84/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#feed-forward-model/dense_84/BiasAddBiasAdd,feed-forward-model/dense_84/MatMul:product:0:feed-forward-model/dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,feed-forward-model/dense_84/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^feed-forward-model/dense_80/BiasAdd/ReadVariableOp2^feed-forward-model/dense_80/MatMul/ReadVariableOp3^feed-forward-model/dense_81/BiasAdd/ReadVariableOp2^feed-forward-model/dense_81/MatMul/ReadVariableOp3^feed-forward-model/dense_82/BiasAdd/ReadVariableOp2^feed-forward-model/dense_82/MatMul/ReadVariableOp3^feed-forward-model/dense_83/BiasAdd/ReadVariableOp2^feed-forward-model/dense_83/MatMul/ReadVariableOp3^feed-forward-model/dense_84/BiasAdd/ReadVariableOp2^feed-forward-model/dense_84/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2h
2feed-forward-model/dense_80/BiasAdd/ReadVariableOp2feed-forward-model/dense_80/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_80/MatMul/ReadVariableOp1feed-forward-model/dense_80/MatMul/ReadVariableOp2h
2feed-forward-model/dense_81/BiasAdd/ReadVariableOp2feed-forward-model/dense_81/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_81/MatMul/ReadVariableOp1feed-forward-model/dense_81/MatMul/ReadVariableOp2h
2feed-forward-model/dense_82/BiasAdd/ReadVariableOp2feed-forward-model/dense_82/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_82/MatMul/ReadVariableOp1feed-forward-model/dense_82/MatMul/ReadVariableOp2h
2feed-forward-model/dense_83/BiasAdd/ReadVariableOp2feed-forward-model/dense_83/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_83/MatMul/ReadVariableOp1feed-forward-model/dense_83/MatMul/ReadVariableOp2h
2feed-forward-model/dense_84/BiasAdd/ReadVariableOp2feed-forward-model/dense_84/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_84/MatMul/ReadVariableOp1feed-forward-model/dense_84/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_17
�
�
*__inference_dense_84_layer_call_fn_3781663

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
E__inference_dense_84_layer_call_and_return_conditional_losses_3780957o
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
�
�
__inference_loss_fn_3_3781717L
:dense_83_kernel_regularizer_square_readvariableop_resource:
identity��1dense_83/kernel/Regularizer/Square/ReadVariableOp�
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_83_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_83/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_83/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp
�
�
*__inference_dense_83_layer_call_fn_3781637

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
E__inference_dense_83_layer_call_and_return_conditional_losses_3780941o
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3781576

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�P
�
 __inference__traced_save_3781857
file_prefix.
*savev2_dense_80_kernel_read_readvariableop,
(savev2_dense_80_bias_read_readvariableop.
*savev2_dense_81_kernel_read_readvariableop,
(savev2_dense_81_bias_read_readvariableop.
*savev2_dense_82_kernel_read_readvariableop,
(savev2_dense_82_bias_read_readvariableop.
*savev2_dense_83_kernel_read_readvariableop,
(savev2_dense_83_bias_read_readvariableop.
*savev2_dense_84_kernel_read_readvariableop,
(savev2_dense_84_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_80_kernel_m_read_readvariableop3
/savev2_adam_dense_80_bias_m_read_readvariableop5
1savev2_adam_dense_81_kernel_m_read_readvariableop3
/savev2_adam_dense_81_bias_m_read_readvariableop5
1savev2_adam_dense_82_kernel_m_read_readvariableop3
/savev2_adam_dense_82_bias_m_read_readvariableop5
1savev2_adam_dense_83_kernel_m_read_readvariableop3
/savev2_adam_dense_83_bias_m_read_readvariableop5
1savev2_adam_dense_84_kernel_m_read_readvariableop3
/savev2_adam_dense_84_bias_m_read_readvariableop5
1savev2_adam_dense_80_kernel_v_read_readvariableop3
/savev2_adam_dense_80_bias_v_read_readvariableop5
1savev2_adam_dense_81_kernel_v_read_readvariableop3
/savev2_adam_dense_81_bias_v_read_readvariableop5
1savev2_adam_dense_82_kernel_v_read_readvariableop3
/savev2_adam_dense_82_bias_v_read_readvariableop5
1savev2_adam_dense_83_kernel_v_read_readvariableop3
/savev2_adam_dense_83_bias_v_read_readvariableop5
1savev2_adam_dense_84_kernel_v_read_readvariableop3
/savev2_adam_dense_84_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_80_kernel_read_readvariableop(savev2_dense_80_bias_read_readvariableop*savev2_dense_81_kernel_read_readvariableop(savev2_dense_81_bias_read_readvariableop*savev2_dense_82_kernel_read_readvariableop(savev2_dense_82_bias_read_readvariableop*savev2_dense_83_kernel_read_readvariableop(savev2_dense_83_bias_read_readvariableop*savev2_dense_84_kernel_read_readvariableop(savev2_dense_84_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_80_kernel_m_read_readvariableop/savev2_adam_dense_80_bias_m_read_readvariableop1savev2_adam_dense_81_kernel_m_read_readvariableop/savev2_adam_dense_81_bias_m_read_readvariableop1savev2_adam_dense_82_kernel_m_read_readvariableop/savev2_adam_dense_82_bias_m_read_readvariableop1savev2_adam_dense_83_kernel_m_read_readvariableop/savev2_adam_dense_83_bias_m_read_readvariableop1savev2_adam_dense_84_kernel_m_read_readvariableop/savev2_adam_dense_84_bias_m_read_readvariableop1savev2_adam_dense_80_kernel_v_read_readvariableop/savev2_adam_dense_80_bias_v_read_readvariableop1savev2_adam_dense_81_kernel_v_read_readvariableop/savev2_adam_dense_81_bias_v_read_readvariableop1savev2_adam_dense_82_kernel_v_read_readvariableop/savev2_adam_dense_82_bias_v_read_readvariableop1savev2_adam_dense_83_kernel_v_read_readvariableop/savev2_adam_dense_83_bias_v_read_readvariableop1savev2_adam_dense_84_kernel_v_read_readvariableop/savev2_adam_dense_84_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�;
�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3780988

inputs"
dense_80_3780873:
dense_80_3780875:"
dense_81_3780896:
dense_81_3780898:"
dense_82_3780919:
dense_82_3780921:"
dense_83_3780942:
dense_83_3780944:"
dense_84_3780958:
dense_84_3780960:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/Square/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/Square/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/Square/ReadVariableOp� dense_83/StatefulPartitionedCall�1dense_83/kernel/Regularizer/Square/ReadVariableOp� dense_84/StatefulPartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinputsdense_80_3780873dense_80_3780875*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3780872�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_3780896dense_81_3780898*
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
E__inference_dense_81_layer_call_and_return_conditional_losses_3780895�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_3780919dense_82_3780921*
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
E__inference_dense_82_layer_call_and_return_conditional_losses_3780918�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_3780942dense_83_3780944*
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
E__inference_dense_83_layer_call_and_return_conditional_losses_3780941�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_3780958dense_84_3780960*
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
E__inference_dense_84_layer_call_and_return_conditional_losses_3780957�
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_3780873*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_81_3780896*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_82_3780919*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_83_3780942*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_84/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/Square/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/Square/ReadVariableOp!^dense_83/StatefulPartitionedCall2^dense_83/kernel/Regularizer/Square/ReadVariableOp!^dense_84/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_82_layer_call_fn_3781611

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
E__inference_dense_82_layer_call_and_return_conditional_losses_3780918o
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

�
4__inference_feed-forward-model_layer_call_fn_3781011
input_17
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
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3780988o
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
input_17
�

�
4__inference_feed-forward-model_layer_call_fn_3781189
input_17
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
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781141o
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
input_17
�
�
*__inference_dense_81_layer_call_fn_3781585

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
E__inference_dense_81_layer_call_and_return_conditional_losses_3780895o
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781242
input_17"
dense_80_3781192:
dense_80_3781194:"
dense_81_3781197:
dense_81_3781199:"
dense_82_3781202:
dense_82_3781204:"
dense_83_3781207:
dense_83_3781209:"
dense_84_3781212:
dense_84_3781214:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/Square/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/Square/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/Square/ReadVariableOp� dense_83/StatefulPartitionedCall�1dense_83/kernel/Regularizer/Square/ReadVariableOp� dense_84/StatefulPartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinput_17dense_80_3781192dense_80_3781194*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3780872�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_3781197dense_81_3781199*
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
E__inference_dense_81_layer_call_and_return_conditional_losses_3780895�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_3781202dense_82_3781204*
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
E__inference_dense_82_layer_call_and_return_conditional_losses_3780918�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_3781207dense_83_3781209*
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
E__inference_dense_83_layer_call_and_return_conditional_losses_3780941�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_3781212dense_84_3781214*
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
E__inference_dense_84_layer_call_and_return_conditional_losses_3780957�
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_3781192*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_81_3781197*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_82_3781202*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_83_3781207*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_84/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/Square/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/Square/ReadVariableOp!^dense_83/StatefulPartitionedCall2^dense_83/kernel/Regularizer/Square/ReadVariableOp!^dense_84/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_17
�
�
__inference_loss_fn_1_3781695L
:dense_81_kernel_regularizer_square_readvariableop_resource:
identity��1dense_81/kernel/Regularizer/Square/ReadVariableOp�
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_81_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_81/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp
�

�
4__inference_feed-forward-model_layer_call_fn_3781426

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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781141o
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
�<
�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781295
input_17"
dense_80_3781245:
dense_80_3781247:"
dense_81_3781250:
dense_81_3781252:"
dense_82_3781255:
dense_82_3781257:"
dense_83_3781260:
dense_83_3781262:"
dense_84_3781265:
dense_84_3781267:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/Square/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/Square/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/Square/ReadVariableOp� dense_83/StatefulPartitionedCall�1dense_83/kernel/Regularizer/Square/ReadVariableOp� dense_84/StatefulPartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinput_17dense_80_3781245dense_80_3781247*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3780872�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_3781250dense_81_3781252*
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
E__inference_dense_81_layer_call_and_return_conditional_losses_3780895�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_3781255dense_82_3781257*
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
E__inference_dense_82_layer_call_and_return_conditional_losses_3780918�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_3781260dense_83_3781262*
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
E__inference_dense_83_layer_call_and_return_conditional_losses_3780941�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_3781265dense_84_3781267*
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
E__inference_dense_84_layer_call_and_return_conditional_losses_3780957�
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_3781245*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_81_3781250*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_82_3781255*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_83_3781260*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_84/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/Square/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/Square/ReadVariableOp!^dense_83/StatefulPartitionedCall2^dense_83/kernel/Regularizer/Square/ReadVariableOp!^dense_84/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
input_17
�L
�	
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781550

inputs9
'dense_80_matmul_readvariableop_resource:6
(dense_80_biasadd_readvariableop_resource:9
'dense_81_matmul_readvariableop_resource:6
(dense_81_biasadd_readvariableop_resource:9
'dense_82_matmul_readvariableop_resource:6
(dense_82_biasadd_readvariableop_resource:9
'dense_83_matmul_readvariableop_resource:6
(dense_83_biasadd_readvariableop_resource:9
'dense_84_matmul_readvariableop_resource:6
(dense_84_biasadd_readvariableop_resource:
identity��dense_80/BiasAdd/ReadVariableOp�dense_80/MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/Square/ReadVariableOp�dense_81/BiasAdd/ReadVariableOp�dense_81/MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/Square/ReadVariableOp�dense_82/BiasAdd/ReadVariableOp�dense_82/MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/Square/ReadVariableOp�dense_83/BiasAdd/ReadVariableOp�dense_83/MatMul/ReadVariableOp�1dense_83/kernel/Regularizer/Square/ReadVariableOp�dense_84/BiasAdd/ReadVariableOp�dense_84/MatMul/ReadVariableOp�
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_80/MatMulMatMulinputs&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_81/MatMulMatMuldense_80/Relu:activations:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_82/MatMulMatMuldense_81/Relu:activations:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_84/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/Square/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp2^dense_83/kernel/Regularizer/Square/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
%__inference_signature_wrapper_3781352
input_17
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
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
"__inference__wrapped_model_3780848o
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
input_17
ț
�
#__inference__traced_restore_3781984
file_prefix2
 assignvariableop_dense_80_kernel:.
 assignvariableop_1_dense_80_bias:4
"assignvariableop_2_dense_81_kernel:.
 assignvariableop_3_dense_81_bias:4
"assignvariableop_4_dense_82_kernel:.
 assignvariableop_5_dense_82_bias:4
"assignvariableop_6_dense_83_kernel:.
 assignvariableop_7_dense_83_bias:4
"assignvariableop_8_dense_84_kernel:.
 assignvariableop_9_dense_84_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: <
*assignvariableop_19_adam_dense_80_kernel_m:6
(assignvariableop_20_adam_dense_80_bias_m:<
*assignvariableop_21_adam_dense_81_kernel_m:6
(assignvariableop_22_adam_dense_81_bias_m:<
*assignvariableop_23_adam_dense_82_kernel_m:6
(assignvariableop_24_adam_dense_82_bias_m:<
*assignvariableop_25_adam_dense_83_kernel_m:6
(assignvariableop_26_adam_dense_83_bias_m:<
*assignvariableop_27_adam_dense_84_kernel_m:6
(assignvariableop_28_adam_dense_84_bias_m:<
*assignvariableop_29_adam_dense_80_kernel_v:6
(assignvariableop_30_adam_dense_80_bias_v:<
*assignvariableop_31_adam_dense_81_kernel_v:6
(assignvariableop_32_adam_dense_81_bias_v:<
*assignvariableop_33_adam_dense_82_kernel_v:6
(assignvariableop_34_adam_dense_82_bias_v:<
*assignvariableop_35_adam_dense_83_kernel_v:6
(assignvariableop_36_adam_dense_83_bias_v:<
*assignvariableop_37_adam_dense_84_kernel_v:6
(assignvariableop_38_adam_dense_84_bias_v:
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
AssignVariableOpAssignVariableOp assignvariableop_dense_80_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_80_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_81_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_81_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_82_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_82_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_83_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_83_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_84_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_84_biasIdentity_9:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_80_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_80_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_81_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_81_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_82_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_82_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_83_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_83_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_84_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_84_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_80_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_80_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_81_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_81_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_82_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_82_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_83_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_83_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_84_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_84_bias_vIdentity_38:output:0"/device:CPU:0*
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
�
�
*__inference_dense_80_layer_call_fn_3781559

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
E__inference_dense_80_layer_call_and_return_conditional_losses_3780872o
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3780872

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_83_layer_call_and_return_conditional_losses_3781654

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_83/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_83/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781141

inputs"
dense_80_3781091:
dense_80_3781093:"
dense_81_3781096:
dense_81_3781098:"
dense_82_3781101:
dense_82_3781103:"
dense_83_3781106:
dense_83_3781108:"
dense_84_3781111:
dense_84_3781113:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/Square/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/Square/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/Square/ReadVariableOp� dense_83/StatefulPartitionedCall�1dense_83/kernel/Regularizer/Square/ReadVariableOp� dense_84/StatefulPartitionedCall�
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinputsdense_80_3781091dense_80_3781093*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3780872�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0dense_81_3781096dense_81_3781098*
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
E__inference_dense_81_layer_call_and_return_conditional_losses_3780895�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0dense_82_3781101dense_82_3781103*
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
E__inference_dense_82_layer_call_and_return_conditional_losses_3780918�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0dense_83_3781106dense_83_3781108*
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
E__inference_dense_83_layer_call_and_return_conditional_losses_3780941�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall)dense_83/StatefulPartitionedCall:output:0dense_84_3781111dense_84_3781113*
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
E__inference_dense_84_layer_call_and_return_conditional_losses_3780957�
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_3781091*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_81_3781096*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_82_3781101*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_83_3781106*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_84/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/Square/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/Square/ReadVariableOp!^dense_83/StatefulPartitionedCall2^dense_83/kernel/Regularizer/Square/ReadVariableOp!^dense_84/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_84_layer_call_and_return_conditional_losses_3781673

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
�L
�	
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781488

inputs9
'dense_80_matmul_readvariableop_resource:6
(dense_80_biasadd_readvariableop_resource:9
'dense_81_matmul_readvariableop_resource:6
(dense_81_biasadd_readvariableop_resource:9
'dense_82_matmul_readvariableop_resource:6
(dense_82_biasadd_readvariableop_resource:9
'dense_83_matmul_readvariableop_resource:6
(dense_83_biasadd_readvariableop_resource:9
'dense_84_matmul_readvariableop_resource:6
(dense_84_biasadd_readvariableop_resource:
identity��dense_80/BiasAdd/ReadVariableOp�dense_80/MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/Square/ReadVariableOp�dense_81/BiasAdd/ReadVariableOp�dense_81/MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/Square/ReadVariableOp�dense_82/BiasAdd/ReadVariableOp�dense_82/MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/Square/ReadVariableOp�dense_83/BiasAdd/ReadVariableOp�dense_83/MatMul/ReadVariableOp�1dense_83/kernel/Regularizer/Square/ReadVariableOp�dense_84/BiasAdd/ReadVariableOp�dense_84/MatMul/ReadVariableOp�
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_80/MatMulMatMulinputs&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_81/MatMulMatMuldense_80/Relu:activations:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_82/MatMulMatMuldense_81/Relu:activations:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_83/MatMulMatMuldense_82/Relu:activations:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_83/ReluReludense_83/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_84/MatMulMatMuldense_83/Relu:activations:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1dense_83/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_83/kernel/Regularizer/SquareSquare9dense_83/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_83/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_83/kernel/Regularizer/SumSum&dense_83/kernel/Regularizer/Square:y:0*dense_83/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_83/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_83/kernel/Regularizer/mulMul*dense_83/kernel/Regularizer/mul/x:output:0(dense_83/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_84/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/Square/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp2^dense_83/kernel/Regularizer/Square/ReadVariableOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp2f
1dense_83/kernel/Regularizer/Square/ReadVariableOp1dense_83/kernel/Regularizer/Square/ReadVariableOp2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_82_layer_call_and_return_conditional_losses_3781628

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_3781684L
:dense_80_kernel_regularizer_square_readvariableop_resource:
identity��1dense_80/kernel/Regularizer/Square/ReadVariableOp�
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_80_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_80/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp
�	
�
E__inference_dense_84_layer_call_and_return_conditional_losses_3780957

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
�

�
4__inference_feed-forward-model_layer_call_fn_3781401

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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3780988o
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
�
�
E__inference_dense_81_layer_call_and_return_conditional_losses_3781602

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_82_layer_call_and_return_conditional_losses_3780918

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/Square/ReadVariableOpt
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
1dense_82/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_82/kernel/Regularizer/SquareSquare9dense_82/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_82/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_82/kernel/Regularizer/SumSum&dense_82/kernel/Regularizer/Square:y:0*dense_82/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �A�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0(dense_82/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/Square/ReadVariableOp1dense_82/kernel/Regularizer/Square/ReadVariableOp:O K
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
input_171
serving_default_input_17:0���������<
dense_840
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
4__inference_feed-forward-model_layer_call_fn_3781011
4__inference_feed-forward-model_layer_call_fn_3781401
4__inference_feed-forward-model_layer_call_fn_3781426
4__inference_feed-forward-model_layer_call_fn_3781189�
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781488
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781550
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781242
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781295�
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
"__inference__wrapped_model_3780848input_17"�
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
*__inference_dense_80_layer_call_fn_3781559�
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3781576�
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
!:2dense_80/kernel
:2dense_80/bias
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
*__inference_dense_81_layer_call_fn_3781585�
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
E__inference_dense_81_layer_call_and_return_conditional_losses_3781602�
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
!:2dense_81/kernel
:2dense_81/bias
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
*__inference_dense_82_layer_call_fn_3781611�
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
E__inference_dense_82_layer_call_and_return_conditional_losses_3781628�
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
!:2dense_82/kernel
:2dense_82/bias
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
*__inference_dense_83_layer_call_fn_3781637�
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
E__inference_dense_83_layer_call_and_return_conditional_losses_3781654�
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
!:2dense_83/kernel
:2dense_83/bias
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
*__inference_dense_84_layer_call_fn_3781663�
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
E__inference_dense_84_layer_call_and_return_conditional_losses_3781673�
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
!:2dense_84/kernel
:2dense_84/bias
�
rtrace_02�
__inference_loss_fn_0_3781684�
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
__inference_loss_fn_1_3781695�
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
__inference_loss_fn_2_3781706�
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
__inference_loss_fn_3_3781717�
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
4__inference_feed-forward-model_layer_call_fn_3781011input_17"�
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
4__inference_feed-forward-model_layer_call_fn_3781401inputs"�
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
4__inference_feed-forward-model_layer_call_fn_3781426inputs"�
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
4__inference_feed-forward-model_layer_call_fn_3781189input_17"�
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781488inputs"�
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781550inputs"�
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781242input_17"�
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781295input_17"�
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
%__inference_signature_wrapper_3781352input_17"�
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
*__inference_dense_80_layer_call_fn_3781559inputs"�
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
E__inference_dense_80_layer_call_and_return_conditional_losses_3781576inputs"�
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
*__inference_dense_81_layer_call_fn_3781585inputs"�
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
E__inference_dense_81_layer_call_and_return_conditional_losses_3781602inputs"�
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
*__inference_dense_82_layer_call_fn_3781611inputs"�
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
E__inference_dense_82_layer_call_and_return_conditional_losses_3781628inputs"�
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
*__inference_dense_83_layer_call_fn_3781637inputs"�
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
E__inference_dense_83_layer_call_and_return_conditional_losses_3781654inputs"�
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
*__inference_dense_84_layer_call_fn_3781663inputs"�
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
E__inference_dense_84_layer_call_and_return_conditional_losses_3781673inputs"�
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
__inference_loss_fn_0_3781684"�
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
__inference_loss_fn_1_3781695"�
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
__inference_loss_fn_2_3781706"�
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
__inference_loss_fn_3_3781717"�
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
&:$2Adam/dense_80/kernel/m
 :2Adam/dense_80/bias/m
&:$2Adam/dense_81/kernel/m
 :2Adam/dense_81/bias/m
&:$2Adam/dense_82/kernel/m
 :2Adam/dense_82/bias/m
&:$2Adam/dense_83/kernel/m
 :2Adam/dense_83/bias/m
&:$2Adam/dense_84/kernel/m
 :2Adam/dense_84/bias/m
&:$2Adam/dense_80/kernel/v
 :2Adam/dense_80/bias/v
&:$2Adam/dense_81/kernel/v
 :2Adam/dense_81/bias/v
&:$2Adam/dense_82/kernel/v
 :2Adam/dense_82/bias/v
&:$2Adam/dense_83/kernel/v
 :2Adam/dense_83/bias/v
&:$2Adam/dense_84/kernel/v
 :2Adam/dense_84/bias/v�
"__inference__wrapped_model_3780848t
&'./671�.
'�$
"�
input_17���������
� "3�0
.
dense_84"�
dense_84����������
E__inference_dense_80_layer_call_and_return_conditional_losses_3781576\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_80_layer_call_fn_3781559O/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_81_layer_call_and_return_conditional_losses_3781602\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_81_layer_call_fn_3781585O/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_82_layer_call_and_return_conditional_losses_3781628\&'/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_82_layer_call_fn_3781611O&'/�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_83_layer_call_and_return_conditional_losses_3781654\.//�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_83_layer_call_fn_3781637O.//�,
%�"
 �
inputs���������
� "�����������
E__inference_dense_84_layer_call_and_return_conditional_losses_3781673\67/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� }
*__inference_dense_84_layer_call_fn_3781663O67/�,
%�"
 �
inputs���������
� "�����������
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781242n
&'./679�6
/�,
"�
input_17���������
p 

 
� "%�"
�
0���������
� �
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781295n
&'./679�6
/�,
"�
input_17���������
p

 
� "%�"
�
0���������
� �
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781488l
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
O__inference_feed-forward-model_layer_call_and_return_conditional_losses_3781550l
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
4__inference_feed-forward-model_layer_call_fn_3781011a
&'./679�6
/�,
"�
input_17���������
p 

 
� "�����������
4__inference_feed-forward-model_layer_call_fn_3781189a
&'./679�6
/�,
"�
input_17���������
p

 
� "�����������
4__inference_feed-forward-model_layer_call_fn_3781401_
&'./677�4
-�*
 �
inputs���������
p 

 
� "�����������
4__inference_feed-forward-model_layer_call_fn_3781426_
&'./677�4
-�*
 �
inputs���������
p

 
� "����������<
__inference_loss_fn_0_3781684�

� 
� "� <
__inference_loss_fn_1_3781695�

� 
� "� <
__inference_loss_fn_2_3781706&�

� 
� "� <
__inference_loss_fn_3_3781717.�

� 
� "� �
%__inference_signature_wrapper_3781352�
&'./67=�:
� 
3�0
.
input_17"�
input_17���������"3�0
.
dense_84"�
dense_84���������