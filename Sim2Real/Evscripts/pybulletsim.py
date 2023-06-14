import pybullet as p
import pybullet_data
import time
import pandas as pd
import numpy as np
# Start the simulation
client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used to locate .urdf and other model files

# Load the UR5 model
robot = p.loadURDF("/home/moga/Desktop/IR-DRL/modular_drl_env/assets/robots/predefined/ur5/urdf/ur5.urdf",baseOrientation=p.getQuaternionFromEuler([0,0,-np.pi]),basePosition=[0,0,0.01])

# Set the gravity
p.setGravity(0, 0, -9.8, physicsClientId=client)

# Define joint angles
joint_angles = [[-3.1415958, -0.78545552, -1.5707396, -2.356148, 1.5707563, -5.991e-05],
[-3.0374436, -0.82216626, -1.4949745, -2.6064632, 1.3205377, -5.991e-05],
[-2.9210389, -0.85651809, -1.4270786, -2.8563964, 1.070695, -4.799e-05],
[-2.7894361, -0.88070089, -1.3934606, -3.1078799, 0.81902856, -7.183e-05],
[-2.7221401, -0.88247329, -1.4081286, -3.1327498, 0.69583857, -4.799e-05],
[-2.630477, -0.87943107, -1.4479831, -3.1400492, 0.5628702, -2.367e-05],
[-2.5436573, -0.86926204, -1.51418817, -3.14147544, 0.39224368, 0.00010786],
[-2.4871602, -0.86293775, -1.5589861, -3.1416552, 0.27669442, -3.559e-05],
[-2.4006798, -0.85847026, -1.6123708, -3.1415951, 0.16239749, -0.04702598],
[-2.3085349, -0.853679, -1.6753396, -3.1416314, -0.00476867, -0.11468775],
[-2.262655, -0.8519781, -1.7062124, -3.1293457, -0.10273582, -0.11792547],
[-2.187919, -0.8600157, -1.7162756, -3.1460063, -0.22226506, -0.11820156],
[-2.136985, -0.88061696, -1.664821, -3.143129, -0.35358125, -0.11828548],
[-2.085488, -0.8966668, -1.6269916, -3.141919, -0.4336961, -0.11833317],
[-2.0063694, -0.9168127, -1.5853008, -3.141691, -0.5858925, -0.14922363],
[-1.959905, -0.9336775, -1.5416425, -3.1199372, -0.67301637, -0.16384059],
[-1.8944782, -0.9638737, -1.4556111, -3.1112592, -0.7952159, -0.16747409],
[-1.8322982, -0.97394735, -1.4349221, -3.019046, -0.860915, -0.16805011],
[-1.735967, -1.0012449, -1.363489, -2.891042, -0.8787158, -0.1683014],
[-1.6743616, -1.0286626, -1.3043388, -2.9328058, -1.0204667, -0.1681698],
[-1.6397079, -1.0461267, -1.2603363, -2.886368, -1.0616931, -0.16820604],
[-1.6125644, -1.0619258, -1.2700027, -2.9088857, -1.2079738, -0.16822988],
[-1.586511, -1.051145, -1.3638252, -2.874636, -1.3906735, -0.16820604],
[-1.5483707, -1.0375861, -1.451917, -2.7800872, -1.4811453, -0.16825372],
[-1.5223173, -1.0273327, -1.5579184, -2.6822612, -1.5010842, -0.16820604],
[-1.537722, -1.0482587, -1.6349319, -2.5244143, -1.4077581, -0.16822988],
[-1.5728554, -1.077617, -1.6702179, -2.5019982, -1.3501133, -0.16820604],
[-1.5902241, -1.0931286, -1.6862186, -2.4984038, -1.3226227, -0.16819412],
[-1.6274894, -1.1331729, -1.7279819, -2.488867, -1.2500538, -0.16820604],
[-1.6560344, -1.1648682, -1.7632207, -2.4774258, -1.1822704, -0.16818172],
[-1.6803507, -1.1977848, -1.7957848, -2.4634917, -1.112882, -0.16821796],
[-1.7016009, -1.2319006, -1.8254818, -2.4492228, -1.042848, -0.16822988],
[-1.7210416, -1.2676219, -1.8525285, -2.4357088, -0.97176105, -0.16818172],
[-1.7385544, -1.3030561, -1.8757852, -2.4248426, -0.90187484, -0.16818172],
[-1.8002075, -1.3491768, -1.8003181, -2.442993, -0.7741435, -0.16820604],
[-1.7906607, -1.4083084, -1.6716455, -2.618613, -0.79984266, -0.16821796],
[-1.7702855, -1.4474462, -1.5842808, -2.7137148, -0.85446626, -0.16819412],
[-1.8293751, -1.5251502, -1.430365, -2.901408, -0.87380105, -0.16821796],
[-1.8521703, -1.6093148, -1.2546037, -3.110864, -0.9166191, -0.16819412],
[-1.885806, -1.6774877, -1.114807, -3.1416433, -1.0049547, -0.16819412],
[-1.87336, -1.7192081, -1.0438274, -3.1415832, -1.0885104, -0.16815788],
[-1.8572253, -1.7768406, -0.976435, -3.1417272, -1.145862, -0.16818172],
[-1.888142, -1.7961797, -1.0282143, -3.1413076, -1.2854451, -0.16820604],
[-1.9009227, -1.7838019, -1.1286701, -3.1416194, -1.3259915, -0.16820604],
[-1.8969458, -1.7791413, -1.1674169, -3.1415832, -1.28867, -0.1681698],
[-1.892753, -1.7686809, -1.2614397, -3.1415832, -1.2127212, -0.16820604],
[-1.9011744, -1.7659367, -1.3218607, -3.0792105, -1.1169943, -0.16819412],
[-1.9384755, -1.7772483, -1.3326901, -3.141907, -1.0035161, -0.1681698],
[-1.9635342, -1.826711, -1.2359666, -3.1000054, -0.83720523, -0.16821796],
[-1.9893359, -1.8482555, -1.2420954, -2.963761, -0.701279, -0.16821796],
[-2.0125382, -1.8515629, -1.3020958, -2.9336445, -0.58969194, -0.16820604],
[-2.0116398, -1.8497051, -1.3372358, -2.989, -0.5315636, -0.16820604],
[-2.004441, -1.8467335, -1.4244035, -3.1167371, -0.4441946, -0.16820604],
[-1.9714288, -1.9100507, -1.2806042, -3.1485593, -0.44418222, -0.16819412],
[-1.9617137, -1.9471625, -1.1909703, -3.1415355, -0.52002174, -0.16820604],
[-1.940751, -1.980812, -1.1084756, -3.141607, -0.5171092, -0.16821796],
[-1.9465011, -1.9349397, -1.2224993, -3.1414754, -0.59238845, -0.16821796],
[-1.9593776, -1.9347118, -1.2559108, -3.1415355, -0.5544432, -0.1681698],
[-1.9959363, -1.9356107, -1.3478016, -3.1416552, -0.46058923, -0.16839725],
[-1.9920195, -1.9609793, -1.2501901, -3.1416314, -0.4189189, -0.19602472],
[-2.0021288, -1.9609073, -1.2776784, -3.1416552, -0.4023326, -0.1959408],
[-3.1415598, -0.78540784, -1.5707277, -2.3561838, 1.5706724, -2.367e-05],
[-3.0373597, -0.81938773, -1.5059007, -2.6062357, 1.3206815, 7.191e-05],
[-2.9212666, -0.85420638, -1.436901, -2.8562648, 1.0706351, 8.389e-05],
[-2.7907181, -0.87687951, -1.4081529, -3.0821712, 0.82074118, 3.595e-05],
[-2.6763976, -0.87790996, -1.4426225, -3.1449273, 0.62152064, -0.02592975],
[-2.6241875, -0.8749636, -1.4706391, -3.1428297, 0.5453684, -0.04998809],
[-2.532564, -0.8657406, -1.5368088, -3.1415832, 0.3982579, -0.08757288],
[-2.487424, -0.86096174, -1.572071, -3.1400015, 0.3071366, -0.09101469],
[-2.3807583, -0.85516435, -1.6413606, -3.118523, 0.14600757, -0.09225018],
[-2.3483424, -0.8497985, -1.6779665, -3.0858386, 0.0540144, -0.09222633],
[-2.289165, -0.84791833, -1.7173313, -3.0461438, -0.03749115, -0.09225018],
[-2.2411292, -0.8438099, -1.762009, -2.9809945, -0.15520793, -0.09225018],
[-2.17788, -0.84501964, -1.7976679, -2.921121, -0.25331384, -0.09225018],
[-2.1377275, -0.84470826, -1.8281087, -2.848152, -0.35842305, -0.09229834],
[-2.062597, -0.85090095, -1.8569545, -2.7624962, -0.47887784, -0.0922621],
[-1.9611863, -0.8837908, -1.7906269, -2.835438, -0.44677097, -0.09223825],
[-1.9298385, -0.9021171, -1.735742, -2.9374917, -0.54878646, -0.0922621],
[-1.860711, -0.93549806, -1.6398252, -3.0357168, -0.6536949, -0.09225018],
[-1.8314594, -0.95205146, -1.5910457, -2.9986837, -0.74378186, -0.0922621],
[-1.772621, -0.9889911, -1.497217, -2.9263103, -0.8553775, -0.09223825],
[-1.7606429, -0.9814933, -1.592101, -2.7596443, -1.0122308, -0.09228642],
[-1.7199997, -0.97340804, -1.6478251, -2.6642997, -1.1018528, -0.09222633],
[-1.6384743, -0.96169406, -1.7630767, -2.5096538, -1.1639999, -0.09231026],
[-1.6258725, -1.0286506, -1.8068794, -2.2525842, -1.0978969, -0.0922621],
[-1.6576995, -1.065267, -1.828133, -2.2601187, -1.057042, -0.09225018],
[-1.6749605, -1.0860976, -1.8419617, -2.262778, -1.0281633, -0.09225018],
[-1.7046794, -1.1307653, -1.8719829, -2.264252, -0.9631422, -0.09225018],
[-1.7298578, -1.1680778, -1.8996528, -2.2608616, -0.8948024, -0.09223825],
[-1.7526053, -1.2066492, -1.9249605, -2.2564533, -0.82363635, -0.0922621],
[-1.787331, -1.2517976, -1.9362472, -2.2136157, -0.7240527, -0.09225018],
[-1.8217214, -1.3138498, -1.9036711, -2.2296674, -0.57610065, -0.09223825],
[-1.8227752, -1.3642591, -1.7974156, -2.3680441, -0.5896681, -0.09222633],
[-1.8212417, -1.4196051, -1.6759275, -2.5214312, -0.64966756, -0.09229834],
[-1.8722585, -1.5062097, -1.505217, -2.7298076, -0.6894124, -0.09222633],
[-1.88807, -1.5874261, -1.3338541, -2.9380906, -0.80666286, -0.09225018],
[-1.9328574, -1.6764215, -1.15389, -3.1464136, -0.8680118, -0.09228642],
[-1.925311, -1.7206215, -1.1254925, -3.141691, -0.94513685, -0.09228642],
[-1.9598569, -1.7213048, -1.1986336, -3.1413195, -1.0713314, -0.09227402],
[-1.9539994, -1.7164997, -1.2394809, -3.1416314, -1.0879225, -0.0922621],
[-1.9451116, -1.7058123, -1.3402103, -3.1416194, -1.0778049, -0.09229834],
[-1.9541429, -1.7052134, -1.3877996, -3.0823987, -0.99742633, -0.09532005],
[-1.993553, -1.7107843, -1.4601091, -2.9935424, -0.8824561, -0.09681923],
[-2.0075314, -1.7130855, -1.5066084, -2.9344714, -0.7926386, -0.09681923],
[-2.0194857, -1.7206091, -1.6098641, -2.8271215, -0.6311019, -0.09684307],
[-2.0015657, -1.7503604, -1.5396994, -2.7020676, -0.50616723, -0.09681923],
[-2.0207317, -1.8375548, -1.440967, -3.0186026, -0.4540456, -0.09687931],
[-2.0513852, -1.8552531, -1.4796709, -3.071936, -0.38167173, -0.09683115],
[-2.0143592, -1.8514069, -1.4192709, -3.0234082, -0.2566331, -0.09683115],
[-2.0103586, -1.940008, -1.2376221, -3.156985, -0.2926076, -0.09683115],
[-1.9981645, -1.956138, -1.1979622, -3.146282, -0.35010606, -0.09687931],
[-1.9741596, -1.9682893, -1.1216787, -3.141607, -0.4598344, -0.09681923],
[-1.9930733, -1.9547237, -1.2143797, -3.1415355, -0.5911301, -0.09687931],
[-2.0087411, -1.9613026, -1.2539438, -3.1415951, -0.4204653, -0.09687931],
[-2.0317643, -1.9520038, -1.3028392, -3.1000292, -0.30310518, -0.09687931],
[-3.1416075, -0.78535968, -1.5708002, -2.35622, 1.5707324, -2.367e-05],
[-3.0376353, -0.8216036, -1.497517, -2.6061876, 1.3207654, -2.367e-05],
[-2.9212666, -0.85642225, -1.4279536, -2.8562167, 1.0705512, 1.198e-05],
[-2.7903464, -0.87884408, -1.3994693, -3.0799539, 0.82063329, -3.559e-05],
[-2.6758704, -0.87941915, -1.435006, -3.142338, 0.62104166, -0.02402288],
[-2.6232526, -0.87632877, -1.4629873, -3.1418707, 0.54389507, -0.04594642],
[-2.5292215, -0.86684257, -1.5307282, -3.1414273, 0.3915134, -0.07996971],
[-2.48383, -0.8618, -1.5664343, -3.138347, 0.29859447, -0.08187658],
[-2.3794165, -0.8556798, -1.6348718, -3.1136572, 0.14248495, -0.08253653],
[-2.3462095, -0.8501461, -1.6724485, -3.0793426, 0.04899376, -0.08258421],
[-2.270322, -0.8477023, -1.7225965, -3.0273993, -0.06586535, -0.08257229],
[-2.2392247, -0.8441213, -1.7559642, -2.9746668, -0.15608293, -0.08256037],
[-2.1642368, -0.8453191, -1.7989391, -2.900605, -0.27421266, -0.08256037],
[-2.1347568, -0.84463626, -1.8236464, -2.836744, -0.36278468, -0.08256037],
[-2.0607283, -0.85056525, -1.8524326, -2.7499864, -0.48236543, -0.08256037],
[-2.0199893, -0.85412246, -1.8705796, -2.6524136, -0.60740644, -0.08256037],
[-1.9514121, -0.8646148, -1.8796953, -2.5557096, -0.7091659, -0.08259613],
[-1.8538474, -0.88117963, -1.883653, -2.4466355, -0.77355605, -0.08252413],
[-1.7832102, -0.8936723, -1.8846368, -2.339412, -0.82762796, -0.08256037],
[-1.7046794, -0.90874034, -1.8782562, -2.2406404, -0.886699, -0.08257229],
[-1.6183141, -0.92687446, -1.861884, -2.1593387, -0.9547866, -0.08258421],
[-1.5692738, -0.96161014, -1.8312267, -2.119785, -1.0303687, -0.08254845],
[-1.6046222, -1.0225896, -1.8190655, -2.166694, -1.0587558, -0.08254845],
[-1.6444153, -1.0617937, -1.835977, -2.2001994, -1.0278634, -0.08254845],
[-1.6796201, -1.0990824, -1.862604, -2.215628, -0.9726723, -0.08258421],
[-1.7091955, -1.1361912, -1.8911976, -2.2188146, -0.9061187, -0.08257229],
[-1.7349972, -1.1742953, -1.9178361, -2.2182395, -0.8353351, -0.08256037],
[-1.7587141, -1.2139925, -1.940997, -2.219928, -0.76307994, -0.08256037],
[-1.8047236, -1.2474135, -1.9266638, -2.1641781, -0.68364745, -0.08258421],
[-1.8989938, -1.3129514, -1.8629516, -2.1395261, -0.62405425, -0.08258421],
[-1.8880938, -1.3708595, -1.8389515, -2.1395142, -0.46852332, -0.08258421],
[-1.879589, -1.4369401, -1.7049292, -2.3480375, -0.5724209, -0.08257229],
[-1.9237417, -1.4994644, -1.6607784, -2.5979922, -0.6558526, -0.08254845],
[-1.967523, -1.5397667, -1.5959991, -2.7047517, -0.57023937, -0.08254845],
[-1.966936, -1.6043547, -1.4612483, -2.8479364, -0.5896438, -0.08256037],
[-1.9514602, -1.6235842, -1.4566301, -3.0144436, -0.7475694, -0.08254845],
[-1.9734043, -1.7101978, -1.3544701, -3.1415231, -0.8222936, -0.08257229],
[-1.9844369, -1.7943105, -1.283027, -3.1415355, -0.77031976, -0.08258421],
[-1.9919475, -1.799044, -1.3241876, -3.0940006, -0.67347175, -0.08254845],
[-2.0255353, -1.8108586, -1.3825587, -3.0912921, -0.52036935, -0.08257229],
[-1.9976853, -1.8964018, -1.2162508, -3.147876, -0.4649518, -0.1224106],
[-2.0612795, -1.9330701, -1.2214559, -3.1416194, -0.5387066, -0.12245829],
[-2.0135686, -2.0059054, -1.1536978, -3.1416552, -0.42644483, -0.12244637],
[-2.0203483, -1.9783434, -1.2250055, -3.1415713, -0.38376933, -0.12242252],
[-2.045192, -1.9426807, -1.3236955, -3.0999577, -0.30293733, -0.12245829],
[-2.0800622, -1.9603561, -1.2841543, -3.0499551, -0.12103302, -0.12244637],
[-2.0920165, -1.9634241, -1.3086079, -3.094684, 0.04556702, -0.12243444]]

# Initialize an array to store end effector positions
end_effector_positions = []

joints = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
joint_ids = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

for angles in joint_angles:
    
    for idx,i in enumerate(joint_ids):
        p.resetJointState(robot, i, angles[idx])
   
    
    # Calculate the end effector position
    ls = p.getLinkState(robot, 7)  # Link 7 is usually the end effector
    end_effector_positions.append(ls[4])  # ls[4] contains the Cartesian coordinates
    
    # Add some delay for visual purposes
    time.sleep(0.05)

# Print the end effector positions
for i, pos in enumerate(end_effector_positions):
    print(f"End effector position for joint angles set {i+1}: {pos}")
#TODO: add end effector position to csv 

# End the simulation
p.disconnect()


# Your pybullet script
# ... your code above

# Convert the tuples in end_effector_positions to lists (vectors) and then to strings
end_effector_positions_str = ['[' + ' '.join(map(str, pos)) + ']' for pos in end_effector_positions]
# Convert end_effector_positions_str list to a DataFrame
df_end_effector = pd.DataFrame(end_effector_positions_str, columns=['real_ee_position'])

# Load the existing CSV data
df = pd.read_csv("/home/moga/Desktop/IR-DRL/Sim2Real/EvSpecial/episode_real_2.csv")

# Check if the size of df_end_effector is the same as df. If not, there might be some issue.
if len(df) != len(df_end_effector):
    print("Warning: the size of the end effector positions data and the original DataFrame is not equal!")

# Concatenate the end_effector_positions to the right of df (i.e., column-wise)
df = pd.concat([df, df_end_effector], axis=1)

# Save the DataFrame to CSV, overwrite the original file
df.to_csv("/home/moga/Desktop/IR-DRL/Sim2Real/EvSpecial/episode_real_2.csv", index=False)
