кв
И▌
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
delete_old_dirsbool(И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018Я■
М
SGD/dense_18/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameSGD/dense_18/bias/momentum
Е
.SGD/dense_18/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_18/bias/momentum*
_output_shapes
:
*
dtype0
Ф
SGD/dense_18/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*-
shared_nameSGD/dense_18/kernel/momentum
Н
0SGD/dense_18/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_18/kernel/momentum*
_output_shapes

:d
*
dtype0
М
SGD/dense_17/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*+
shared_nameSGD/dense_17/bias/momentum
Е
.SGD/dense_17/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_17/bias/momentum*
_output_shapes
:d*
dtype0
Ф
SGD/dense_17/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*-
shared_nameSGD/dense_17/kernel/momentum
Н
0SGD/dense_17/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_17/kernel/momentum*
_output_shapes

:dd*
dtype0
М
SGD/dense_16/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*+
shared_nameSGD/dense_16/bias/momentum
Е
.SGD/dense_16/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_16/bias/momentum*
_output_shapes
:d*
dtype0
Ф
SGD/dense_16/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*-
shared_nameSGD/dense_16/kernel/momentum
Н
0SGD/dense_16/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_16/kernel/momentum*
_output_shapes

:dd*
dtype0
М
SGD/dense_15/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*+
shared_nameSGD/dense_15/bias/momentum
Е
.SGD/dense_15/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_15/bias/momentum*
_output_shapes
:d*
dtype0
Ф
SGD/dense_15/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*-
shared_nameSGD/dense_15/kernel/momentum
Н
0SGD/dense_15/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_15/kernel/momentum*
_output_shapes

:dd*
dtype0
М
SGD/dense_14/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*+
shared_nameSGD/dense_14/bias/momentum
Е
.SGD/dense_14/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_14/bias/momentum*
_output_shapes
:d*
dtype0
Х
SGD/dense_14/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Рd*-
shared_nameSGD/dense_14/kernel/momentum
О
0SGD/dense_14/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_14/kernel/momentum*
_output_shapes
:	Рd*
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
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:
*
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:d
*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:d*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:dd*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:d*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:dd*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:d*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:dd*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:d*
dtype0
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Рd* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	Рd*
dtype0

NoOpNoOp
┌:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Х:
valueЛ:BИ: BБ:
Ь
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
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
ж
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
ж
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
ж
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
* 
░
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
=trace_0
>trace_1
?trace_2
@trace_3* 
6
Atrace_0
Btrace_1
Ctrace_2
Dtrace_3* 
* 
▄
Eiter
	Fdecay
Glearning_rate
Hmomentummomentumxmomentumymomentumzmomentum{&momentum|'momentum}.momentum~/momentum6momentumА7momentumБ*

Iserving_default* 

0
1*

0
1*
* 
У
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Otrace_0* 

Ptrace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
У
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Vtrace_0* 

Wtrace_0* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 
У
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
_Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
У
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
_Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
У
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
_Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_18/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

m0
n1*
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
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
* 
* 
* 
* 
* 
8
o	variables
p	keras_api
	qtotal
	rcount*
H
s	variables
t	keras_api
	utotal
	vcount
w
_fn_kwargs*

q0
r1*

o	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

s	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
РЙ
VARIABLE_VALUESGD/dense_14/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUESGD/dense_14/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUESGD/dense_15/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUESGD/dense_15/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUESGD/dense_16/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUESGD/dense_16/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUESGD/dense_17/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUESGD/dense_17/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUESGD/dense_18/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUESGD/dense_18/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_input_5Placeholder*(
_output_shapes
:         Р*
dtype0*
shape:         Р
ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5dense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_60026645
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Б
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/dense_14/kernel/momentum/Read/ReadVariableOp.SGD/dense_14/bias/momentum/Read/ReadVariableOp0SGD/dense_15/kernel/momentum/Read/ReadVariableOp.SGD/dense_15/bias/momentum/Read/ReadVariableOp0SGD/dense_16/kernel/momentum/Read/ReadVariableOp.SGD/dense_16/bias/momentum/Read/ReadVariableOp0SGD/dense_17/kernel/momentum/Read/ReadVariableOp.SGD/dense_17/bias/momentum/Read/ReadVariableOp0SGD/dense_18/kernel/momentum/Read/ReadVariableOp.SGD/dense_18/bias/momentum/Read/ReadVariableOpConst*)
Tin"
 2	*
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
GPU 2J 8В **
f%R#
!__inference__traced_save_60026980
╠
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_1count_1totalcountSGD/dense_14/kernel/momentumSGD/dense_14/bias/momentumSGD/dense_15/kernel/momentumSGD/dense_15/bias/momentumSGD/dense_16/kernel/momentumSGD/dense_16/bias/momentumSGD/dense_17/kernel/momentumSGD/dense_17/bias/momentumSGD/dense_18/kernel/momentumSGD/dense_18/bias/momentum*(
Tin!
2*
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_60027074Иў
░

№
5__inference_feed-forward-model_layer_call_fn_60026556
input_5
unknown:	Рd
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:d

	unknown_8:

identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         Р
!
_user_specified_name	input_5
д>
ъ
!__inference__traced_save_60026980
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_sgd_dense_14_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_14_bias_momentum_read_readvariableop;
7savev2_sgd_dense_15_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_15_bias_momentum_read_readvariableop;
7savev2_sgd_dense_16_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_16_bias_momentum_read_readvariableop;
7savev2_sgd_dense_17_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_17_bias_momentum_read_readvariableop;
7savev2_sgd_dense_18_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_18_bias_momentum_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╞
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*я
valueхBтB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHз
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╪
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_dense_14_kernel_momentum_read_readvariableop5savev2_sgd_dense_14_bias_momentum_read_readvariableop7savev2_sgd_dense_15_kernel_momentum_read_readvariableop5savev2_sgd_dense_15_bias_momentum_read_readvariableop7savev2_sgd_dense_16_kernel_momentum_read_readvariableop5savev2_sgd_dense_16_bias_momentum_read_readvariableop7savev2_sgd_dense_17_kernel_momentum_read_readvariableop5savev2_sgd_dense_17_bias_momentum_read_readvariableop7savev2_sgd_dense_18_kernel_momentum_read_readvariableop5savev2_sgd_dense_18_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*╦
_input_shapes╣
╢: :	Рd:d:dd:d:dd:d:dd:d:d
:
: : : : : : : : :	Рd:d:dd:d:dd:d:dd:d:d
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Рd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$	 

_output_shapes

:d
: 


_output_shapes
:
:
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
: :%!

_output_shapes
:	Рd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:

_output_shapes
: 
б

°
F__inference_dense_14_layer_call_and_return_conditional_losses_60026793

inputs1
matmul_readvariableop_resource:	Рd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Рd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Р: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
Э

ў
F__inference_dense_16_layer_call_and_return_conditional_losses_60026338

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Є,
■
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026773

inputs:
'dense_14_matmul_readvariableop_resource:	Рd6
(dense_14_biasadd_readvariableop_resource:d9
'dense_15_matmul_readvariableop_resource:dd6
(dense_15_biasadd_readvariableop_resource:d9
'dense_16_matmul_readvariableop_resource:dd6
(dense_16_biasadd_readvariableop_resource:d9
'dense_17_matmul_readvariableop_resource:dd6
(dense_17_biasadd_readvariableop_resource:d9
'dense_18_matmul_readvariableop_resource:d
6
(dense_18_biasadd_readvariableop_resource:

identityИвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвdense_16/BiasAdd/ReadVariableOpвdense_16/MatMul/ReadVariableOpвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpвdense_18/BiasAdd/ReadVariableOpвdense_18/MatMul/ReadVariableOpЗ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	Рd*
dtype0{
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Р
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Р
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Р
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0Р
dense_18/MatMulMatMuldense_17/Relu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Д
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_18/SoftmaxSoftmaxdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_18/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Х
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
н

√
5__inference_feed-forward-model_layer_call_fn_60026670

inputs
unknown:	Рd
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:d

	unknown_8:

identityИвStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
Э

ў
F__inference_dense_17_layer_call_and_return_conditional_losses_60026355

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Э

ў
F__inference_dense_17_layer_call_and_return_conditional_losses_60026853

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Э

ў
F__inference_dense_16_layer_call_and_return_conditional_losses_60026833

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
в

ў
F__inference_dense_18_layer_call_and_return_conditional_losses_60026372

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
░

№
5__inference_feed-forward-model_layer_call_fn_60026402
input_5
unknown:	Рd
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:d

	unknown_8:

identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026379o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         Р
!
_user_specified_name	input_5
°<
╬

#__inference__wrapped_model_60026286
input_5M
:feed_forward_model_dense_14_matmul_readvariableop_resource:	РdI
;feed_forward_model_dense_14_biasadd_readvariableop_resource:dL
:feed_forward_model_dense_15_matmul_readvariableop_resource:ddI
;feed_forward_model_dense_15_biasadd_readvariableop_resource:dL
:feed_forward_model_dense_16_matmul_readvariableop_resource:ddI
;feed_forward_model_dense_16_biasadd_readvariableop_resource:dL
:feed_forward_model_dense_17_matmul_readvariableop_resource:ddI
;feed_forward_model_dense_17_biasadd_readvariableop_resource:dL
:feed_forward_model_dense_18_matmul_readvariableop_resource:d
I
;feed_forward_model_dense_18_biasadd_readvariableop_resource:

identityИв2feed-forward-model/dense_14/BiasAdd/ReadVariableOpв1feed-forward-model/dense_14/MatMul/ReadVariableOpв2feed-forward-model/dense_15/BiasAdd/ReadVariableOpв1feed-forward-model/dense_15/MatMul/ReadVariableOpв2feed-forward-model/dense_16/BiasAdd/ReadVariableOpв1feed-forward-model/dense_16/MatMul/ReadVariableOpв2feed-forward-model/dense_17/BiasAdd/ReadVariableOpв1feed-forward-model/dense_17/MatMul/ReadVariableOpв2feed-forward-model/dense_18/BiasAdd/ReadVariableOpв1feed-forward-model/dense_18/MatMul/ReadVariableOpн
1feed-forward-model/dense_14/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_14_matmul_readvariableop_resource*
_output_shapes
:	Рd*
dtype0в
"feed-forward-model/dense_14/MatMulMatMulinput_59feed-forward-model/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dк
2feed-forward-model/dense_14/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╩
#feed-forward-model/dense_14/BiasAddBiasAdd,feed-forward-model/dense_14/MatMul:product:0:feed-forward-model/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dИ
 feed-forward-model/dense_14/ReluRelu,feed-forward-model/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         dм
1feed-forward-model/dense_15/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_15_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0╔
"feed-forward-model/dense_15/MatMulMatMul.feed-forward-model/dense_14/Relu:activations:09feed-forward-model/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dк
2feed-forward-model/dense_15/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_15_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╩
#feed-forward-model/dense_15/BiasAddBiasAdd,feed-forward-model/dense_15/MatMul:product:0:feed-forward-model/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dИ
 feed-forward-model/dense_15/ReluRelu,feed-forward-model/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         dм
1feed-forward-model/dense_16/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_16_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0╔
"feed-forward-model/dense_16/MatMulMatMul.feed-forward-model/dense_15/Relu:activations:09feed-forward-model/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dк
2feed-forward-model/dense_16/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_16_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╩
#feed-forward-model/dense_16/BiasAddBiasAdd,feed-forward-model/dense_16/MatMul:product:0:feed-forward-model/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dИ
 feed-forward-model/dense_16/ReluRelu,feed-forward-model/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:         dм
1feed-forward-model/dense_17/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_17_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0╔
"feed-forward-model/dense_17/MatMulMatMul.feed-forward-model/dense_16/Relu:activations:09feed-forward-model/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dк
2feed-forward-model/dense_17/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_17_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╩
#feed-forward-model/dense_17/BiasAddBiasAdd,feed-forward-model/dense_17/MatMul:product:0:feed-forward-model/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dИ
 feed-forward-model/dense_17/ReluRelu,feed-forward-model/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         dм
1feed-forward-model/dense_18/MatMul/ReadVariableOpReadVariableOp:feed_forward_model_dense_18_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0╔
"feed-forward-model/dense_18/MatMulMatMul.feed-forward-model/dense_17/Relu:activations:09feed-forward-model/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
к
2feed-forward-model/dense_18/BiasAdd/ReadVariableOpReadVariableOp;feed_forward_model_dense_18_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╩
#feed-forward-model/dense_18/BiasAddBiasAdd,feed-forward-model/dense_18/MatMul:product:0:feed-forward-model/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
О
#feed-forward-model/dense_18/SoftmaxSoftmax,feed-forward-model/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:         
|
IdentityIdentity-feed-forward-model/dense_18/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
╙
NoOpNoOp3^feed-forward-model/dense_14/BiasAdd/ReadVariableOp2^feed-forward-model/dense_14/MatMul/ReadVariableOp3^feed-forward-model/dense_15/BiasAdd/ReadVariableOp2^feed-forward-model/dense_15/MatMul/ReadVariableOp3^feed-forward-model/dense_16/BiasAdd/ReadVariableOp2^feed-forward-model/dense_16/MatMul/ReadVariableOp3^feed-forward-model/dense_17/BiasAdd/ReadVariableOp2^feed-forward-model/dense_17/MatMul/ReadVariableOp3^feed-forward-model/dense_18/BiasAdd/ReadVariableOp2^feed-forward-model/dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 2h
2feed-forward-model/dense_14/BiasAdd/ReadVariableOp2feed-forward-model/dense_14/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_14/MatMul/ReadVariableOp1feed-forward-model/dense_14/MatMul/ReadVariableOp2h
2feed-forward-model/dense_15/BiasAdd/ReadVariableOp2feed-forward-model/dense_15/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_15/MatMul/ReadVariableOp1feed-forward-model/dense_15/MatMul/ReadVariableOp2h
2feed-forward-model/dense_16/BiasAdd/ReadVariableOp2feed-forward-model/dense_16/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_16/MatMul/ReadVariableOp1feed-forward-model/dense_16/MatMul/ReadVariableOp2h
2feed-forward-model/dense_17/BiasAdd/ReadVariableOp2feed-forward-model/dense_17/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_17/MatMul/ReadVariableOp1feed-forward-model/dense_17/MatMul/ReadVariableOp2h
2feed-forward-model/dense_18/BiasAdd/ReadVariableOp2feed-forward-model/dense_18/BiasAdd/ReadVariableOp2f
1feed-forward-model/dense_18/MatMul/ReadVariableOp1feed-forward-model/dense_18/MatMul/ReadVariableOp:Q M
(
_output_shapes
:         Р
!
_user_specified_name	input_5
Ф
■
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026585
input_5$
dense_14_60026559:	Рd
dense_14_60026561:d#
dense_15_60026564:dd
dense_15_60026566:d#
dense_16_60026569:dd
dense_16_60026571:d#
dense_17_60026574:dd
dense_17_60026576:d#
dense_18_60026579:d

dense_18_60026581:

identityИв dense_14/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв dense_16/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallў
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_14_60026559dense_14_60026561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_60026304Щ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_60026564dense_15_60026566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_60026321Щ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_60026569dense_16_60026571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_60026338Щ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_60026574dense_17_60026576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_60026355Щ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_60026579dense_18_60026581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_60026372x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ї
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:Q M
(
_output_shapes
:         Р
!
_user_specified_name	input_5
╞
Ш
+__inference_dense_18_layer_call_fn_60026862

inputs
unknown:d

	unknown_0:

identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_60026372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
н

√
5__inference_feed-forward-model_layer_call_fn_60026695

inputs
unknown:	Рd
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:d

	unknown_8:

identityИвStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
С
¤
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026379

inputs$
dense_14_60026305:	Рd
dense_14_60026307:d#
dense_15_60026322:dd
dense_15_60026324:d#
dense_16_60026339:dd
dense_16_60026341:d#
dense_17_60026356:dd
dense_17_60026358:d#
dense_18_60026373:d

dense_18_60026375:

identityИв dense_14/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв dense_16/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallЎ
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_60026305dense_14_60026307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_60026304Щ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_60026322dense_15_60026324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_60026321Щ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_60026339dense_16_60026341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_60026338Щ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_60026356dense_17_60026358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_60026355Щ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_60026373dense_18_60026375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_60026372x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ї
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
С
¤
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026508

inputs$
dense_14_60026482:	Рd
dense_14_60026484:d#
dense_15_60026487:dd
dense_15_60026489:d#
dense_16_60026492:dd
dense_16_60026494:d#
dense_17_60026497:dd
dense_17_60026499:d#
dense_18_60026502:d

dense_18_60026504:

identityИв dense_14/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв dense_16/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallЎ
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_60026482dense_14_60026484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_60026304Щ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_60026487dense_15_60026489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_60026321Щ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_60026492dense_16_60026494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_60026338Щ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_60026497dense_17_60026499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_60026355Щ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_60026502dense_18_60026504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_60026372x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ї
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
╞
Ш
+__inference_dense_17_layer_call_fn_60026842

inputs
unknown:dd
	unknown_0:d
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_60026355o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
дq
ї
$__inference__traced_restore_60027074
file_prefix3
 assignvariableop_dense_14_kernel:	Рd.
 assignvariableop_1_dense_14_bias:d4
"assignvariableop_2_dense_15_kernel:dd.
 assignvariableop_3_dense_15_bias:d4
"assignvariableop_4_dense_16_kernel:dd.
 assignvariableop_5_dense_16_bias:d4
"assignvariableop_6_dense_17_kernel:dd.
 assignvariableop_7_dense_17_bias:d4
"assignvariableop_8_dense_18_kernel:d
.
 assignvariableop_9_dense_18_bias:
&
assignvariableop_10_sgd_iter:	 '
assignvariableop_11_sgd_decay: /
%assignvariableop_12_sgd_learning_rate: *
 assignvariableop_13_sgd_momentum: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: C
0assignvariableop_18_sgd_dense_14_kernel_momentum:	Рd<
.assignvariableop_19_sgd_dense_14_bias_momentum:dB
0assignvariableop_20_sgd_dense_15_kernel_momentum:dd<
.assignvariableop_21_sgd_dense_15_bias_momentum:dB
0assignvariableop_22_sgd_dense_16_kernel_momentum:dd<
.assignvariableop_23_sgd_dense_16_bias_momentum:dB
0assignvariableop_24_sgd_dense_17_kernel_momentum:dd<
.assignvariableop_25_sgd_dense_17_bias_momentum:dB
0assignvariableop_26_sgd_dense_18_kernel_momentum:d
<
.assignvariableop_27_sgd_dense_18_bias_momentum:

identity_29ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╔
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*я
valueхBтB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHк
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ░
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*И
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_15_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_15_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_16_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_16_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_17_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_17_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_18_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_18_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:Н
AssignVariableOp_10AssignVariableOpassignvariableop_10_sgd_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp%assignvariableop_12_sgd_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_13AssignVariableOp assignvariableop_13_sgd_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_18AssignVariableOp0assignvariableop_18_sgd_dense_14_kernel_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_19AssignVariableOp.assignvariableop_19_sgd_dense_14_bias_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_20AssignVariableOp0assignvariableop_20_sgd_dense_15_kernel_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_21AssignVariableOp.assignvariableop_21_sgd_dense_15_bias_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_22AssignVariableOp0assignvariableop_22_sgd_dense_16_kernel_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_23AssignVariableOp.assignvariableop_23_sgd_dense_16_bias_momentumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_24AssignVariableOp0assignvariableop_24_sgd_dense_17_kernel_momentumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_25AssignVariableOp.assignvariableop_25_sgd_dense_17_bias_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_26AssignVariableOp0assignvariableop_26_sgd_dense_18_kernel_momentumIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_27AssignVariableOp.assignvariableop_27_sgd_dense_18_bias_momentumIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╖
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: д
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
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
Э

ў
F__inference_dense_15_layer_call_and_return_conditional_losses_60026813

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Є,
■
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026734

inputs:
'dense_14_matmul_readvariableop_resource:	Рd6
(dense_14_biasadd_readvariableop_resource:d9
'dense_15_matmul_readvariableop_resource:dd6
(dense_15_biasadd_readvariableop_resource:d9
'dense_16_matmul_readvariableop_resource:dd6
(dense_16_biasadd_readvariableop_resource:d9
'dense_17_matmul_readvariableop_resource:dd6
(dense_17_biasadd_readvariableop_resource:d9
'dense_18_matmul_readvariableop_resource:d
6
(dense_18_biasadd_readvariableop_resource:

identityИвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвdense_16/BiasAdd/ReadVariableOpвdense_16/MatMul/ReadVariableOpвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpвdense_18/BiasAdd/ReadVariableOpвdense_18/MatMul/ReadVariableOpЗ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	Рd*
dtype0{
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Р
dense_15/MatMulMatMuldense_14/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Р
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Р
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         dЖ
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0Р
dense_18/MatMulMatMuldense_17/Relu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Д
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_18/SoftmaxSoftmaxdense_18/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_18/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Х
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
╞
Ш
+__inference_dense_15_layer_call_fn_60026802

inputs
unknown:dd
	unknown_0:d
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_60026321o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ї	
э
&__inference_signature_wrapper_60026645
input_5
unknown:	Рd
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:dd
	unknown_6:d
	unknown_7:d

	unknown_8:

identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_60026286o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         Р
!
_user_specified_name	input_5
б

°
F__inference_dense_14_layer_call_and_return_conditional_losses_60026304

inputs1
matmul_readvariableop_resource:	Рd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Рd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Р: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
Э

ў
F__inference_dense_15_layer_call_and_return_conditional_losses_60026321

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╞
Ш
+__inference_dense_16_layer_call_fn_60026822

inputs
unknown:dd
	unknown_0:d
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_60026338o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
в

ў
F__inference_dense_18_layer_call_and_return_conditional_losses_60026873

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╔
Щ
+__inference_dense_14_layer_call_fn_60026782

inputs
unknown:	Рd
	unknown_0:d
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_60026304o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Р: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Р
 
_user_specified_nameinputs
Ф
■
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026614
input_5$
dense_14_60026588:	Рd
dense_14_60026590:d#
dense_15_60026593:dd
dense_15_60026595:d#
dense_16_60026598:dd
dense_16_60026600:d#
dense_17_60026603:dd
dense_17_60026605:d#
dense_18_60026608:d

dense_18_60026610:

identityИв dense_14/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв dense_16/StatefulPartitionedCallв dense_17/StatefulPartitionedCallв dense_18/StatefulPartitionedCallў
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_5dense_14_60026588dense_14_60026590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_60026304Щ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_60026593dense_15_60026595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_60026321Щ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_60026598dense_16_60026600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_60026338Щ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_60026603dense_17_60026605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_60026355Щ
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_60026608dense_18_60026610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_60026372x
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ї
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         Р: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall:Q M
(
_output_shapes
:         Р
!
_user_specified_name	input_5"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*м
serving_defaultШ
<
input_51
serving_default_input_5:0         Р<
dense_180
StatefulPartitionedCall:0         
tensorflow/serving/predict:ЪУ
│
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
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
╗
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
╗
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
 "
trackable_list_wrapper
╩
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
К
=trace_0
>trace_1
?trace_2
@trace_32Я
5__inference_feed-forward-model_layer_call_fn_60026402
5__inference_feed-forward-model_layer_call_fn_60026670
5__inference_feed-forward-model_layer_call_fn_60026695
5__inference_feed-forward-model_layer_call_fn_60026556└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 z=trace_0z>trace_1z?trace_2z@trace_3
Ў
Atrace_0
Btrace_1
Ctrace_2
Dtrace_32Л
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026734
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026773
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026585
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026614└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zAtrace_0zBtrace_1zCtrace_2zDtrace_3
╬B╦
#__inference__wrapped_model_60026286input_5"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы
Eiter
	Fdecay
Glearning_rate
Hmomentummomentumxmomentumymomentumzmomentum{&momentum|'momentum}.momentum~/momentum6momentumА7momentumБ"
	optimizer
,
Iserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
я
Otrace_02╥
+__inference_dense_14_layer_call_fn_60026782в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zOtrace_0
К
Ptrace_02э
F__inference_dense_14_layer_call_and_return_conditional_losses_60026793в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zPtrace_0
": 	Рd2dense_14/kernel
:d2dense_14/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
я
Vtrace_02╥
+__inference_dense_15_layer_call_fn_60026802в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zVtrace_0
К
Wtrace_02э
F__inference_dense_15_layer_call_and_return_conditional_losses_60026813в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zWtrace_0
!:dd2dense_15/kernel
:d2dense_15/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
я
]trace_02╥
+__inference_dense_16_layer_call_fn_60026822в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z]trace_0
К
^trace_02э
F__inference_dense_16_layer_call_and_return_conditional_losses_60026833в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z^trace_0
!:dd2dense_16/kernel
:d2dense_16/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
я
dtrace_02╥
+__inference_dense_17_layer_call_fn_60026842в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zdtrace_0
К
etrace_02э
F__inference_dense_17_layer_call_and_return_conditional_losses_60026853в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zetrace_0
!:dd2dense_17/kernel
:d2dense_17/bias
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
н
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
я
ktrace_02╥
+__inference_dense_18_layer_call_fn_60026862в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zktrace_0
К
ltrace_02э
F__inference_dense_18_layer_call_and_return_conditional_losses_60026873в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zltrace_0
!:d
2dense_18/kernel
:
2dense_18/bias
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
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ИBЕ
5__inference_feed-forward-model_layer_call_fn_60026402input_5"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЗBД
5__inference_feed-forward-model_layer_call_fn_60026670inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЗBД
5__inference_feed-forward-model_layer_call_fn_60026695inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ИBЕ
5__inference_feed-forward-model_layer_call_fn_60026556input_5"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
вBЯ
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026734inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
вBЯ
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026773inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
гBа
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026585input_5"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
гBа
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026614input_5"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
═B╩
&__inference_signature_wrapper_60026645input_5"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_dense_14_layer_call_fn_60026782inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_14_layer_call_and_return_conditional_losses_60026793inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_dense_15_layer_call_fn_60026802inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_15_layer_call_and_return_conditional_losses_60026813inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_dense_16_layer_call_fn_60026822inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_16_layer_call_and_return_conditional_losses_60026833inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_dense_17_layer_call_fn_60026842inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_17_layer_call_and_return_conditional_losses_60026853inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_dense_18_layer_call_fn_60026862inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_18_layer_call_and_return_conditional_losses_60026873inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
N
o	variables
p	keras_api
	qtotal
	rcount"
_tf_keras_metric
^
s	variables
t	keras_api
	utotal
	vcount
w
_fn_kwargs"
_tf_keras_metric
.
q0
r1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
.
u0
v1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
-:+	Рd2SGD/dense_14/kernel/momentum
&:$d2SGD/dense_14/bias/momentum
,:*dd2SGD/dense_15/kernel/momentum
&:$d2SGD/dense_15/bias/momentum
,:*dd2SGD/dense_16/kernel/momentum
&:$d2SGD/dense_16/bias/momentum
,:*dd2SGD/dense_17/kernel/momentum
&:$d2SGD/dense_17/bias/momentum
,:*d
2SGD/dense_18/kernel/momentum
&:$
2SGD/dense_18/bias/momentumЫ
#__inference__wrapped_model_60026286t
&'./671в.
'в$
"К
input_5         Р
к "3к0
.
dense_18"К
dense_18         
з
F__inference_dense_14_layer_call_and_return_conditional_losses_60026793]0в-
&в#
!К
inputs         Р
к "%в"
К
0         d
Ъ 
+__inference_dense_14_layer_call_fn_60026782P0в-
&в#
!К
inputs         Р
к "К         dж
F__inference_dense_15_layer_call_and_return_conditional_losses_60026813\/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ ~
+__inference_dense_15_layer_call_fn_60026802O/в,
%в"
 К
inputs         d
к "К         dж
F__inference_dense_16_layer_call_and_return_conditional_losses_60026833\&'/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ ~
+__inference_dense_16_layer_call_fn_60026822O&'/в,
%в"
 К
inputs         d
к "К         dж
F__inference_dense_17_layer_call_and_return_conditional_losses_60026853\.//в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ ~
+__inference_dense_17_layer_call_fn_60026842O.//в,
%в"
 К
inputs         d
к "К         dж
F__inference_dense_18_layer_call_and_return_conditional_losses_60026873\67/в,
%в"
 К
inputs         d
к "%в"
К
0         

Ъ ~
+__inference_dense_18_layer_call_fn_60026862O67/в,
%в"
 К
inputs         d
к "К         
┬
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026585n
&'./679в6
/в,
"К
input_5         Р
p 

 
к "%в"
К
0         

Ъ ┬
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026614n
&'./679в6
/в,
"К
input_5         Р
p

 
к "%в"
К
0         

Ъ ┴
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026734m
&'./678в5
.в+
!К
inputs         Р
p 

 
к "%в"
К
0         

Ъ ┴
P__inference_feed-forward-model_layer_call_and_return_conditional_losses_60026773m
&'./678в5
.в+
!К
inputs         Р
p

 
к "%в"
К
0         

Ъ Ъ
5__inference_feed-forward-model_layer_call_fn_60026402a
&'./679в6
/в,
"К
input_5         Р
p 

 
к "К         
Ъ
5__inference_feed-forward-model_layer_call_fn_60026556a
&'./679в6
/в,
"К
input_5         Р
p

 
к "К         
Щ
5__inference_feed-forward-model_layer_call_fn_60026670`
&'./678в5
.в+
!К
inputs         Р
p 

 
к "К         
Щ
5__inference_feed-forward-model_layer_call_fn_60026695`
&'./678в5
.в+
!К
inputs         Р
p

 
к "К         
й
&__inference_signature_wrapper_60026645
&'./67<в9
в 
2к/
-
input_5"К
input_5         Р"3к0
.
dense_18"К
dense_18         
