from typing import List,Tuple
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from matplotlib.pyplot import plot,show,figure

def accelerometer():
    for line in accelerometer_data:
        yield tuple(
            map(int,line.strip().strip("()").split(","))
        )

def filter_gravity(accelerometer_readings:List[Tuple[int,int,int]]) -> List[Tuple[int,int,int]]:
    x,y,z = zip(*accelerometer_readings)

    x_biases = [0]
    for x_ in x:
        x_biases.append(
            .9*x_biases[-1] + .1*x_
        )
    y_biases = [0]
    for y_ in y:
        y_biases.append(
            .9*y_biases[-1] + .1*y_
        )
    z_biases = [0]
    for z_ in z:
        z_biases.append(
            .9*z_biases[-1] + .1*z_
        )

    x_unbiased = list(map(lambda x_raw,x_bias:x_raw - x_bias, x,x_biases))
    y_unbiased = list(map(lambda y_raw,y_bias:y_raw - y_bias, y,y_biases))
    z_unbiased = list(map(lambda z_raw,z_bias:z_raw - z_bias, z, z_biases))

    return list(zip(x_unbiased, y_unbiased, z_unbiased))

def normalise(accelerometer_readings:List[Tuple[int,int,int]]) -> List[Tuple[int,int,int]]:
    x,y,z = zip(*accelerometer_readings)
    
    x_median = sum(x)//len(x)
    y_median = sum(y)//len(y)
    z_median = sum(z)//len(z)
    
    x_normalised = list(map(lambda x_raw:x_raw-x_median, x))
    y_normalised = list(map(lambda y_raw:y_raw-y_median, y))
    z_normalised = list(map(lambda z_raw:z_raw-z_median, z))
    return list(zip(x_normalised,z_normalised,y_normalised))

def smooth(accelerometer_readings:List[Tuple[int,int,int]]) -> List[Tuple[int,int,int]]:
    x,y,z = zip(*accelerometer_readings)
    x_smooth = savgol_filter(x=x,window_length=7,polyorder=2,mode='constant')
    y_smooth = savgol_filter(x=y,window_length=7,polyorder=2,mode='constant')
    z_smooth = savgol_filter(x=z,window_length=7,polyorder=2,mode='constant')
    return list(zip(x_smooth,y_smooth,z_smooth))

def double_integrate(accelerometer_readings:List[Tuple[int,int,int]]) -> List[Tuple[int,int,int]]:
    acceleration_x,acceleration_y,acceleration_z = zip(*accelerometer_readings)
    position_x = cumtrapz(cumtrapz(acceleration_x))
    position_y = cumtrapz(cumtrapz(acceleration_y))
    position_z = cumtrapz(cumtrapz(acceleration_z))
    return list(zip(position_x,position_y,position_z))


with open("data/27aug2020.txt") as data:
    accelerometer_data = data.readlines()

raw_acceleration = list(accelerometer())
normalised_raw_acceleration = normalise(accelerometer_readings=raw_acceleration)
normalised_raw_acceleration_without_gravity = filter_gravity(accelerometer_readings=normalised_raw_acceleration)

smooth_acceleration = smooth(accelerometer_readings=raw_acceleration)
normalised_smooth_acceleration = normalise(accelerometer_readings=smooth_acceleration)
normalised_smooth_acceleration_without_gravity = filter_gravity(accelerometer_readings=normalised_smooth_acceleration)

normalised_raw_displacement_without_gravity = double_integrate(normalised_raw_acceleration_without_gravity)
normalised_smooth_displacement_without_gravity = double_integrate(normalised_smooth_acceleration_without_gravity)

plot(normalised_raw_acceleration_without_gravity)
plot(normalised_smooth_acceleration_without_gravity)
show()


fig = figure()
ax = fig.add_subplot(111, projection='3d')
x,y,z = zip(*normalised_raw_displacement_without_gravity)
ax.plot(xs=x,ys=y,zs=z)
x,y,z = zip(*normalised_smooth_displacement_without_gravity)
ax.plot(xs=x,ys=y,zs=z)
show()