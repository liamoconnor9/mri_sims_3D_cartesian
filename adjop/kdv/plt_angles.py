import numpy as np
import matplotlib.pyplot as plt

wavenumbers = list(range(1, 30))

# angles_Tp25 = [3.9253937453117373, 3.4772503998227866, 3.2782080894346532, 3.006694799023987, 2.5554222951124332, 1.9357589576987653, 1.1864996481280585, 0.6568061345971785, 0.6072445706897635, 0.711329356538798, 0.5667038765342528, 0.4837571163772629, 0.5069364943434453, 0.6525056897885418, 0.8704179739624783, 1.4803071246049264, 2.858587708248942, 7.4146972071266894, 40.53857173490041]

# angles_Tp125 = [1.9787586219501039, 1.811714509237827, 1.7575454922317357, 1.713792823409562, 1.6286355402801727, 1.4998758972660753, 1.329991486482292, 1.1244118913070005, 0.8927230622261584, 0.6629208177758826, 0.5037179380664408, 0.46976806767682067, 0.4889392352709382, 0.48674717420835484, 0.5893695674679827, 0.9108470510559243, 1.4232851548553975, 2.414132951672069, 5.712935905712359]

# angles_Tp0625 = [1.6108058224889474, 1.4643102600167492, 1.2653551096993203, 1.7766287261379012, 1.4410956560131152, 1.5645870555113952, 1.5133208326709966, 1.493395519323978, 1.4601776212305533, 1.429452639331732, 1.4085770128377426, 1.3958081490622756, 1.3847638080983866, 1.3889239142652345, 1.4486466881742248, 1.618136191565758, 1.9708806015762588, 2.7073709869942855, 4.532422402760249]

# angles_Tp03125 = [1.0138017783260256, 0.9667418294834762, 0.8840860521222065, 1.0898377488961226, 0.9561805935753825, 1.007740811424192, 0.9941435509998874, 0.9933441677076007, 0.9892010516619112, 0.9837196376643332, 0.9763874348997149, 0.9676536343068867, 0.958750778001538, 0.951370005799045, 0.9472925265659278, 0.9481135827230412, 0.9548448898515748, 0.9704161220739166, 1.0052154301093448]


# plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp03125) / 0.03125, label=r"$T = 0.03125$")
# plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp0625) / 0.0625, label=r"$T = 0.0625$")
# plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp125) / 0.125, label=r"$T = 0.125$")
# plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp25) / 0.25, label=r"$T = 0.250$")
# # plt.scatter(np.array(wavenumbers)**2, np.array(angles_Tp5) / 0.5, label=r"$T = 0.500$")
# plt.legend()
# plt.title('Gradient Error angle ' + r'$\theta$' + '; L2 Norm')
# plt.ylabel(r'$\theta \; / \; T$')
# plt.xlabel('Wavenumber Squared')
# plt.savefig('diff_angles_L2.png')
# plt.show()

# angles_sig0p5 = [2.8903909693754266, 3.0251637085213985, 2.152538403175641, 1.3911358624833283, 1.640939228827782, 1.7494883206029088, 1.3972738603650163, 0.8005037756383685, 0.5984068673867803, 0.4260080300909771, 0.3102670598663749, 0.23004971978075595, 0.18482250179178888, 0.2284225362423796, 0.14512512928945304, 0.15037423457541388, 0.14263813695301902, 0.16850458691177728, 0.2649427749144767, 0.3820335522840748, 0.5989317609181278, 0.9992825747103077, 2.400780682360232, 8.317152609171998, 46.200913728771276, 87.4604560346106, 89.92363876491626, 89.98148811246298, 89.99523388954421]

# angles_sig1p0 = [2.028940175512237, 2.2635433644520475, 1.9349175162295216, 1.9034020095361963, 1.7580604034814948, 1.532403468248396, 1.2537587552633274, 0.9601688045205122, 0.709889494164756, 0.4787284713299276, 0.3020915462986209, 0.21430452560748992, 0.22517588756348672, 0.2858251329267828, 0.20229228586806183, 0.15336831954663144, 0.14551638898092323, 0.16862607882462236, 0.32190102587869385, 0.5344853557836915, 0.8317049703127806, 1.2894146695143351, 2.6591340934470225, 6.641372778653403, 25.50802492478883, 79.2868234112815, 89.85509317037538, 89.99798569503132, 90.00001103009835]

# angles_sig1p5 = [1.9681621865424146, 1.799798349256263, 1.7453618559121937, 1.7013305773509881, 1.6154284864820474, 1.4854231835544816, 1.3137059576780978, 1.1055046546133704, 0.8695050754521765, 0.6307837555980024, 0.45604841946668234, 0.41228217355996494, 0.4548910441654602, 0.47996767504383286, 0.44782088466070175, 0.4328514749184395, 0.4624342230566178, 0.49523421569193077, 0.6915521827889768, 1.0771250880047558, 1.683866806262905, 3.124746975802962, 8.765751014350414, 40.315315068828696, 85.43056280554967, 90.16608850550533, 90.00616502087458, 89.99925188333897, 89.99998543778356]

# plt.scatter(np.array(wavenumbers)**2, np.array(angles_sig0p5), label=r"$\sigma = 0.5$")
# plt.scatter(np.array(wavenumbers)**2, np.array(angles_sig1p0), label=r"$\sigma = 1.0$")
# plt.scatter(np.array(wavenumbers)**2, np.array(angles_sig1p5), label=r"$\sigma = 1.5$")
# plt.legend()
# plt.title('Gradient Error angle ' + r'$\theta$' + '; L2 Norm')
# plt.ylabel(r'$\theta$')
# plt.xlabel('Wavenumber Squared')
# plt.savefig('diff_angles_sig.png')
# plt.show()

angles_coeff0p1 = [0.19607386139981897, 0.18095046377611473, 0.174682852100142, 0.17004919185837974, 0.16158384943263807, 0.14863035785966747, 0.13135880527274954, 0.11040589556322546, 0.08690127001709398, 0.06369069267167321, 0.047683134770261426, 0.04574719702768624, 0.0532526926651361, 0.06075711934804565, 0.06629353074988964, 0.07652965273072289, 0.09232345352433835, 0.11133433796610172, 0.14062049534260895, 0.1834083145928469, 0.24499739627729517, 0.37874489347649226, 0.9373893714712644, 5.164410451229314, 52.43856300500988, 89.28399323213466, 90.00010300102858, 89.99966274626246, 89.99999666550443]

angles_coeff1 = [1.9379924236530097, 1.7658414185838027, 1.702403635341788, 1.6641394172355528, 1.5736734251686613, 1.4399576886730339, 1.2618209104622589, 1.0432672709784587, 0.7897337401510182, 0.519315503171403, 0.2989324482677127, 0.26403645358836364, 0.37721647354310084, 0.47800213541783876, 0.5426394869351279, 0.6535344108917754, 0.8186451612006765, 1.0166283780902923, 1.2781676330453993, 1.6141934086269152, 2.01948694734951, 2.579727000653523, 3.842240558894084, 8.542460509590173, 53.83857767838935, 89.31945095588776, 90.00137973301388, 89.9992433671398, 89.99998506105842]

angles_coeff2 = [3.9036067199864974, 3.5207854013757487, 3.3942724782352878, 3.335377739248648, 3.1408581053018305, 2.8716560767770103, 2.518358149533636, 2.087520123360432, 1.584957908559012, 1.0420195312178206, 0.5962006906256181, 0.5288561349503899, 0.7555246350062257, 0.9581976025069989, 1.0911354773958009, 1.3141540827720055, 1.645390903457455, 2.043048365809034, 2.563067510543989, 3.2278413085568385, 4.021667658276525, 5.09026500721547, 7.378867276312465, 13.93206075373992, 55.992535355159696, 89.33617332311671, 90.0022778329345, 89.99877612937918, 89.99996822264328]

angles_coeff3 = [5.903627866833155, 5.273484239263231, 5.082760151029716, 5.015960028740877, 4.69516995436327, 4.287239002443074, 3.762536783176434, 3.128992693269129, 2.386885218497607, 1.5739047920153617, 0.899349251821334, 0.8008036241989684, 1.138414430354849, 1.442104977367724, 1.6471155432633877, 1.984145305737045, 2.4830852167154704, 3.081215421865917, 3.8583799687098552, 4.852247183442121, 6.03218222595364, 7.601374445342454, 10.895173209494823, 19.374743213633575, 58.27575528482923, 89.33283627857931, 90.00280213604981, 89.99831063622281, 89.9999471240858]

angles_coeff10 = [21.27279485838503, 17.818066588584635, 17.194149439026713, 16.994999747571764, 14.66628756441866, 13.226130085958099, 11.641070490090504, 9.9578737893644, 8.035890546093555, 5.819843489828288, 3.5472063159126237, 3.00636165997621, 3.941262938134475, 4.8766420330856315, 5.676743696760107, 6.901759595278372, 8.640897393721172, 10.622401105494436, 13.02809609851202, 16.216277229026023, 19.84302556876834, 24.229285731185303, 32.118337649892624, 45.478333067211764, 67.26144934998102, 88.98451879068865, 90.0034990542148, 89.9951201483587, 89.99965124636236]

plt.scatter(np.array(wavenumbers)**2, np.array(angles_coeff0p1), label="$u(x, 0)$ coeff = 0.1")
plt.scatter(np.array(wavenumbers)**2, np.array(angles_coeff1), label="$u(x, 0)$ coeff = 1.0")
plt.scatter(np.array(wavenumbers)**2, np.array(angles_coeff2), label="$u(x, 0)$ coeff = 2.0")
plt.scatter(np.array(wavenumbers)**2, np.array(angles_coeff3), label="$u(x, 0)$ coeff = 3.0")
plt.scatter(np.array(wavenumbers)**2, np.array(angles_coeff10), label="$u(x, 0)$ coeff = 10.0")
# plt.scatter(np.array(wavenumbers)**2, np.array(angles_coeff1p1), label="$u(x, 0)$ coeff = 1.1")
plt.legend()
plt.title('Gradient Error angle ' + r'$\theta$' + '; L2 Norm')
plt.ylabel(r'$\theta$')
plt.xlabel('Wavenumber Squared')
plt.savefig('diff_angles_icscale.png')
plt.show()
