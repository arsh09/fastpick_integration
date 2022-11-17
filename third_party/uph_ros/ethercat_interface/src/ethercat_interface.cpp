#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include "ros/ros.h"
#include "ethercat_interface/ethercat_includes.h"
#include <std_msgs/UInt32.h>
#include <ethercat_interface/Cmd.h>
#define EC_TIMEOUTMON 500
#define PDO_PERIOD 5000
#define STATECHECK_PERIOD 100000

typedef struct PACKED
{
	uint16 controlWord;
	int32 TargetPosition;
	int32 TargetVelocity;
}
out_ISMC;

typedef struct PACKED
{
	uint16 statusWord;
	int32 ActPosition;
	int32 DigitalInputs;
	uint8 operation_mode;

}
in_ISMC;

in_ISMC * val1;
out_ISMC * target1;

in_ISMC * val2;
out_ISMC * target2;

typedef enum
{
	NOT_READY, SWITCH_DISABLED, READY_SWITCH, SWITCHED_ON, OPERATION_ENABLED, QUICK_STOP, FAULT_REACTION, FAULT, UNKNOWN
}
PDS_STATUS;	// Statusword(6041h) 

typedef enum
{
	NO_MODE_CHANGE, PROFILE_POSITION_MODE, VELOCITY_MODE, PROFILE_VELOCITY_MODE, TORQUE_PROFILE_MODE, HOMING_MODE, INTERPOLATED_POSITION_MODE, CYCLIC_SYNCHRONOUS_POSITION_MODE, CYCLIC_SYNCHRONOUS_VELOCITY_MODE, CYCLIC_SYNCHRONOUS_TORQUE_MODE
}
PDS_OPERATION;	// Mode of operation(6061h)

char IOmap[4096];
pthread_t thread_statecheck;
pthread_t thread_pdo;

volatile int expectedWKC;
volatile int wkc;
boolean pdo_transfer_active = FALSE;

uint8_t startup_step1 = 0;
uint8_t startup_step2 = 0;
bool homing1 = true;
bool homing2 = true;
bool cmd_executed = false;

/**
 *helper macros
 */
#define READ(slaveId, idx, sub, buf, comment)\
{\
	buf = 0;\
	int __s = sizeof(buf);\
	int __ret = ec_SDOread(slaveId, idx, sub, FALSE, &__s, &buf, EC_TIMEOUTRXM);\
	printf("Slave: %d - Read at 0x%04x:%d => wkc: %d; data: 0x%.*x (%d)\t[%s]\n", slaveId, idx, sub, __ret, __s, (unsigned int) buf, (unsigned int) buf, comment);\
}
#define WRITE(slaveId, idx, sub, buf, value, comment)\
{\
	int __s = sizeof(buf);\
	buf = value;\
	int __ret = ec_SDOwrite(slaveId, idx, sub, FALSE, __s, &buf, EC_TIMEOUTRXM);\
	printf("Slave: %d - Write at 0x%04x:%d => wkc: %d; data: 0x%.*x\t{%s}\n", slaveId, idx, sub, __ret, __s, (unsigned int) buf, comment);\
}

boolean setup_ethercat(char *ifname)
{
	uint32 buf32;
	uint16 buf16;
	uint8 buf8;
	int i, j, chk, cnt;

	/*initialise SOEM, bind socket to ifname */
	if (ec_init(ifname))
	{
		ROS_INFO("ec_init on %s succeeded.", ifname);
		/*find and auto-config slaves */

		if (ec_config_init(FALSE) > 0)
		{
			ROS_INFO("%d slaves found and configured.", ec_slavecount);

			if ((ec_slavecount >= 1))
			{

				for (int i = 1; i <= ec_slavecount; i++)
				{

					WRITE(i, 0x3202, 0, buf32, 8, "Motor Drive Submode Select");
					WRITE(i, 0x2030, 0, buf32, 50, "Pole Pair Count");
					READ(i, 0x6061, 0, buf8, "OpMode display");

					WRITE(i, 0x607B, 1, buf32, -100000, "Min Position Range Limit");
					WRITE(i, 0x607B, 2, buf32, 100000, "Max Position Range Limit");
					WRITE(i, 0x607D, 1, buf32, -100000, "Min Position Limit");
					WRITE(i, 0x607D, 2, buf32, 100000, "Min Position Limit");

					WRITE(i, 0x6082, 0, buf32, 0, "End Velocity");
					WRITE(i, 0x6083, 0, buf32, 500, "Profile Acceleration");
					WRITE(i, 0x6084, 0, buf32, 500, "Profile Deceleration");

				
					if (i == 1)
					{
						WRITE(i, 0x6098, 0, buf8, 17, "Homing Method");
						WRITE(i, 0x6099, 1, buf32, 100, "Homing Speed:Switch");
						WRITE(i, 0x6099, 2, buf32, 100, "Homing Speed:Zero");
						WRITE(i, 0x609A, 0, buf32, 500, "Homing Acceleration");

						WRITE(i, 0x3240, 1, buf32, 1, "Digital Inputs Control:Special function");
						WRITE(i, 0x3240, 2, buf32, 0, "Digital Inputs Control:Inverted");
					}
					if (i == 2)
					{
						WRITE(i, 0x6098, 0, buf8, 17, "Homing Method");
						WRITE(i, 0x6099, 1, buf32, 50, "Homing Speed:Switch");
						WRITE(i, 0x6099, 2, buf32, 50, "Homing Speed:Zero");
						WRITE(i, 0x609A, 0, buf32, 500, "Homing Acceleration");

						WRITE(i, 0x3240, 1, buf32, 1, "Digital Inputs Control:Special function");						
						WRITE(i, 0x3240, 2, buf32, 1, "Digital Inputs Control:Inverted");
					}
					READ(i, 0x1c12, 0, buf32, "rxPDO:0");
					READ(i, 0x1c13, 0, buf32, "txPDO:0");

					READ(i, 0x1c12, 1, buf32, "rxPDO:1");
					READ(i, 0x1c13, 1, buf32, "txPDO:1");

					WRITE(i, 0x1c12, 0, buf8, 0x0, "Rx mapping");
					WRITE(i, 0x1c12, 1, buf16, 0x1601, "Rx mapping");
					WRITE(i, 0x1c12, 0, buf8, 0x1, "Rx mapping");

					WRITE(i, 0x1c13, 0, buf8, 0x0, "Tx mapping");
					WRITE(i, 0x1c13, 1, buf16, 0x1a01, "Rx mapping");
					WRITE(i, 0x1c13, 0, buf8, 0x1, "Tx mapping");

					WRITE(i, 0x1601, 0, buf8, 0x0, "Number of mapped objects");
					WRITE(i, 0x1601, 1, buf32, 0x60400010, "Control word");
					WRITE(i, 0x1601, 2, buf32, 0x607A0020, "Target position");
					WRITE(i, 0x1601, 3, buf32, 0x60810002, "Profile Velocity");
					WRITE(i, 0x1601, 0, buf8, 3, "Number of mapped objects");

					WRITE(i, 0x1a01, 0, buf8, 0x0, "Number of mapped objects");
					WRITE(i, 0x1a01, 1, buf32, 0x60410010, "status word");
					WRITE(i, 0x1a01, 2, buf32, 0x60640020, "position_actual_value;");
					WRITE(i, 0x1a01, 3, buf32, 0x60FD0020, "digital_inputs");
					WRITE(i, 0x1a01, 4, buf32, 0x60610008, "Modes of operation display");
					WRITE(i, 0x1a01, 0, buf8, 0x04, "Number of mapped objects");

					WRITE(i, 0x60c2, 1, buf8, 0x01, "cyc time");
				}
			}

			ec_config_map(&IOmap);
			ec_configdc();

			/*connect struct pointers to slave I/O pointers */
			target1 = (out_ISMC*)(ec_slave[1].outputs);
			val1 = (in_ISMC*)(ec_slave[1].inputs);

			target2 = (out_ISMC*)(ec_slave[2].outputs);
			val2 = (in_ISMC*)(ec_slave[2].inputs);

			/*read indevidual slave state and store in ec_slave[] */
			ec_readstate();
			for (cnt = 1; cnt <= ec_slavecount; cnt++)
			{
				printf("Slave:%d Name:%s Output size:%3dbits Input size:%3dbits State:%2d delay:%d.%d\n",
					cnt, ec_slave[cnt].name, ec_slave[cnt].Obits, ec_slave[cnt].Ibits,
					ec_slave[cnt].state, (int) ec_slave[cnt].pdelay, ec_slave[cnt].hasdc);
			}

			ROS_INFO("Slaves mapped, state to SAFE_OP.");
			/*wait for all slaves to reach SAFE_OP state */
			ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE *4);

			ROS_INFO("segments : %d : %d %d %d %d", ec_group[0].nsegments,
				ec_group[0].IOsegment[0], ec_group[0].IOsegment[1],
				ec_group[0].IOsegment[2], ec_group[0].IOsegment[3]);

			ROS_INFO("Request operational state for all slaves");
			expectedWKC = (ec_group[0].outputsWKC *2) + ec_group[0].inputsWKC;
			ROS_INFO("Calculated workcounter %d", expectedWKC);
			ec_slave[0].state = EC_STATE_OPERATIONAL;
			/*send one valid process data to make outputs in slaves happy*/
			ec_send_processdata();
			ec_receive_processdata(EC_TIMEOUTRET);
			/*request OP state for all slaves */
			ec_writestate(0);
			chk = 40;
			/*wait for all slaves to reach OP state */

			do { 	ec_send_processdata();
				ec_receive_processdata(EC_TIMEOUTRET);
				ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
			} while (chk-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));
			if (ec_slave[0].state == EC_STATE_OPERATIONAL)
			{
				ROS_INFO("Operational state reached for all slaves.");
				pdo_transfer_active = TRUE;
				return TRUE;
			}
			else
			{
				ROS_WARN("Not all slaves reached operational state.");
				ec_readstate();
				for (i = 1; i <= ec_slavecount; i++)
				{
					if (ec_slave[i].state != EC_STATE_OPERATIONAL)
					{
						ROS_WARN("Slave %d State=0x%2.2x StatusCode=0x%4.4x : %s", i,
							ec_slave[i].state, ec_slave[i].ALstatuscode,
							ec_ALstatuscode2string(ec_slave[i].ALstatuscode));
					}
				}
			}
		}
		else
		{
			ROS_ERROR("No slaves found!");
		}
	}
	else
	{
		ROS_ERROR("No socket connection on %s. Try excecuting the following "
			"command: sudo setcap 'cap_net_raw=ep cap_sys_nice=eip' $(readlink $(catkin_find "
			"ethercat_interface ethercat_interface))\n",
			ifname);
	}
	return FALSE;
}

void stop_ethercat()
{
	/*stop PDO transfer in Thread */
	pdo_transfer_active = FALSE;

	/*request INIT state for all slaves */
	ROS_INFO("Request init state for all slaves");
	ec_slave[0].state = EC_STATE_INIT;
	ec_writestate(0);

	/*stop SOEM, close socket */
	ec_close();
}

bool start_soem(char *ifname)
{
	/*initialise SOEM and bring to operational state*/
	uint16 buf16;
	if (setup_ethercat(ifname))
	{
		printf("Setup finished!\n");
		return TRUE;
	}
	else
	{
		ROS_ERROR("Initialization failed");
		return FALSE;
	}
}

void *ecat_pdotransfer(void *ptr)
{
	while (1)
	{
		if (pdo_transfer_active)
		{
			ec_send_processdata();
			wkc = ec_receive_processdata(EC_TIMEOUTRET);
		}
		osal_usleep(PDO_PERIOD);
	}
}

void *ecat_statecheck(void *ptr)
{
	int slave;
	uint8 currentgroup = 0;

	while (1)
	{
		if (pdo_transfer_active &&
			((wkc < expectedWKC) || ec_group[currentgroup].docheckstate))
		{ /*one ore more slaves are not responding */
			ec_group[currentgroup].docheckstate = FALSE;
			ec_readstate();
			for (slave = 1; slave <= ec_slavecount; slave++)
			{
				if ((ec_slave[slave].group == currentgroup) &&
					(ec_slave[slave].state != EC_STATE_OPERATIONAL))
				{
					ec_group[currentgroup].docheckstate = TRUE;
					if (ec_slave[slave].state == (EC_STATE_SAFE_OP + EC_STATE_ERROR))
					{
						ROS_ERROR("ERROR : slave %d is in SAFE_OP + ERROR, attempting ack.",
							slave);
						ec_slave[slave].state = (EC_STATE_SAFE_OP + EC_STATE_ACK);
						ec_writestate(slave);
					}
					else if (ec_slave[slave].state == EC_STATE_SAFE_OP)
					{
						ROS_WARN("slave %d is in SAFE_OP, change to OPERATIONAL.", slave);
						ec_slave[slave].state = EC_STATE_OPERATIONAL;
						ec_writestate(slave);
					}
					else if (ec_slave[slave].state > 0)
					{
						if (ec_reconfig_slave(slave, EC_TIMEOUTMON))
						{
							ec_slave[slave].islost = FALSE;
							ROS_INFO("MESSAGE : slave %d reconfigured", slave);
						}
					}
					else if (!ec_slave[slave].islost)
					{ /*re-check state */
						ec_statecheck(slave, EC_STATE_OPERATIONAL, EC_TIMEOUTRET);
						if (!ec_slave[slave].state)
						{
							ec_slave[slave].islost = TRUE;
							ROS_ERROR("slave %d lost", slave);
						}
					}
				}
				if (ec_slave[slave].islost)
				{
					if (!ec_slave[slave].state)
					{
						if (ec_recover_slave(slave, EC_TIMEOUTMON))
						{
							ec_slave[slave].islost = FALSE;
							ROS_INFO("MESSAGE : slave %d recovered", slave);
						}
					}
					else
					{
						ec_slave[slave].islost = FALSE;
						ROS_INFO("MESSAGE : slave %d found", slave);
					}
				}
			}

			if (!ec_group[currentgroup].docheckstate)
			{
				ROS_INFO("OK : all slaves resumed OPERATIONAL.");
			}
		}
		osal_usleep(STATECHECK_PERIOD);
	}
}

// Based on: http://www.yonch.com/tech/82-linux-thread-priority
void set_realtime_priority(pthread_t *thread)
{
	int ret;
	// struct sched_param is used to store the scheduling priority
	struct sched_param params;

	// We'll set the priority to the maximum.
	params.sched_priority = sched_get_priority_max(SCHED_FIFO);
	ROS_INFO("Trying to set thread realtime prio = %d", params.sched_priority);

	// Attempt to set thread real-time priority to the SCHED_FIFO policy
	ret = pthread_setschedparam(*thread, SCHED_FIFO, &params);
	if (ret != 0)
	{
		ROS_ERROR("Unsuccessful in setting thread realtime prio, got error: %d. Possible errors: ESRCH(%d), EINVAL(%d), EPERM(%d)", ret, ESRCH, EINVAL, EPERM);
		if (ret == EPERM)
		{
			ROS_ERROR("No appropriate permissions. Try excecuting the following "
				"command: sudo setcap 'cap_net_raw=ep cap_sys_nice=eip' $(readlink $(catkin_find "
				"ethercat_interface ethercat_interface))\n");
		}
		return;
	}
	// Now verify the change in thread priority
	int policy = 0;
	ret = pthread_getschedparam(*thread, &policy, &params);
	if (ret != 0)
	{
		ROS_ERROR("Couldn't retrieve real-time scheduling parameters, got error: %d. Possible errors: ESRCH(%d), EINVAL(%d), EPERM(%d)", ret, ESRCH, EINVAL, EPERM);
		return;
	}

	// Check the correct policy was applied
	if (policy != SCHED_FIFO)
	{
		ROS_ERROR("Scheduling is NOT SCHED_FIFO! Got: %d", policy);
	}
	else
	{
		ROS_INFO("SCHED_FIFO OK, Thread priority is %d", params.sched_priority);
	}
}

PDS_STATUS getPDSStatus(int motor)
{
	uint16 status;
	if (motor == 1)
	{
		status = val1->statusWord;
	}
	if (motor == 2)
	{
		status = val2->statusWord;
	}

	if (((status) &0x004f) == 0x0000)
	{
		// x0xx 0000
		return NOT_READY;
	}
	else if (((status) &0x004f) == 0x0040)
	{
		// x1xx 0000
		return SWITCH_DISABLED;
	}
	else if (((status) &0x006f) == 0x0021)
	{
		// x01x 0001
		return READY_SWITCH;
	}
	else if (((status) &0x006f) == 0x0023)
	{
		// x01x 0011
		return SWITCHED_ON;
	}
	else if (((status) &0x006f) == 0x0027)
	{
		// x01x 0111
		return OPERATION_ENABLED;
	}
	else if (((status) &0x006f) == 0x0007)
	{
		// x00x 0111
		return QUICK_STOP;
	}
	else if (((status) &0x004f) == 0x000f)
	{
		// x0xx 1111
		return FAULT_REACTION;
	}
	else if (((status) &0x004f) == 0x0008)
	{
		// x0xx 1000
		return FAULT;
	}
	else
	{
		return UNKNOWN;
	}
}

PDS_OPERATION getPDSOperation(int operation_mode)
{
	switch (operation_mode)
	{
		case 0:
			return NO_MODE_CHANGE;
			break;
		case 1:
			return PROFILE_POSITION_MODE;
			break;	// pp
		case 2:
			return VELOCITY_MODE;
			break;	// vl
		case 3:
			return PROFILE_VELOCITY_MODE;
			break;	// pv
		case 4:
			return TORQUE_PROFILE_MODE;
			break;	// tq
		case 6:
			return HOMING_MODE;
			break;	// hm
		case 7:
			return INTERPOLATED_POSITION_MODE;
			break;	// ip
		case 8:
			return CYCLIC_SYNCHRONOUS_POSITION_MODE;
			break;	// csp
		case 9:
			return CYCLIC_SYNCHRONOUS_VELOCITY_MODE;
			break;	// csv
		case 10:
			return CYCLIC_SYNCHRONOUS_TORQUE_MODE;
			break;	// cst
	}
}

void printPDSStatus(int motor)
{
	uint16 status;
	uint8 mode;
	if (motor == 1)
	{
		status = val1->statusWord;
		mode = val1->operation_mode;
	}
	if (motor == 2)
	{
		status = val2->statusWord;
		mode = val2->operation_mode;
	}
	switch (getPDSStatus(motor))
	{
		case NOT_READY:
			printf(" Not ready to switch on\n");
			break;
		case SWITCH_DISABLED:
			printf(" Switch on disabled\n");
			break;
		case READY_SWITCH:
			printf(" Ready to switch on\n");
			break;
		case SWITCHED_ON:
			printf(" Switched on\n");
			break;
		case OPERATION_ENABLED:
			printf(" Operation enabled\n");
			break;
		case QUICK_STOP:
			printf(" Quick stop active\n");
			break;
		case FAULT_REACTION:
			printf(" Fault reaction active\n");
			break;
		case FAULT:
			printf(" Fault\n");
			break;
		case UNKNOWN:
			printf(" Unknown status %04x\n", status);
			break;
	}
	if (status & 0x0800)
	{
		printf(" Internal limit active\n");
	}
}

void MotorOn(int motor)
{
	uint8 buf8;
	uint16 control;

	while (getPDSStatus(motor) != OPERATION_ENABLED)
	{

		switch (getPDSStatus(motor))
		{
			case SWITCH_DISABLED:
				control = 0x06;	// move to ready to switch on
				printf("   move to ready to switch on\n");
				break;
			case READY_SWITCH:
				control = 0x07;	// move to switched on
				printf("   move to switched on\n");
				break;
			case SWITCHED_ON:
				control = 0x0f;	// move to operation enabled
				printf("   move to operation enabled\n");
				break;
			case OPERATION_ENABLED:
				printf("   Operation enabled\n");
				break;
			default:
				printf("   unknown status\n");
				return;
		}
		if (motor == 1)
		{
			target1->controlWord = control;
		}
		if (motor == 2)
		{
			target2->controlWord = control;
		}
		usleep(1000);
	}
}
void MotorOff(int motor)
{
	uint8 buf8;
	uint16 control;
	while (getPDSStatus(motor) != SWITCH_DISABLED)
	{

		switch (getPDSStatus(motor))
		{
			case READY_SWITCH:
				control = 0x0000;	// disable voltage
				break;
			case SWITCHED_ON:
				control = 0x0006;	// shutdown
				break;
			case OPERATION_ENABLED:
				control = 0x0007;	// disable operation
				break;
			default:
				printf("unknown status");
				control = 0x0000;	// disable operation
				break;
		}
		if (motor == 1)
		{
			target1->controlWord = control;
		}
		if (motor == 2)
		{
			target2->controlWord = control;
		}
		usleep(1000);
	}
}

bool TargetReached(int motor)
{
	uint16 status;	
	if (motor == 1)
	{
		status = val1->statusWord;
	}
	if (motor == 2)
	{
		status = val2->statusWord;
	}
	if(status & 0x0400)
	{
		return true;
	}
	else
	{
		return false;
	}	
}
void NewSetPoint(int motor, bool newsetpoint)
{
	uint16 control;	
	if(newsetpoint)
	{	
		control = control | 0x0010;
		
	}
	else
	{
		control &= ~0x0010; // clear new-set-point (bit4)
		
	}
	if (motor == 1)
	{
		target1->controlWord = control;
	}
	if (motor == 2)
	{
		target2->controlWord = control;
	}
}

//rostopic pub -1 /pos_cmd ethercat_interface/Cmd "{position1: 30000, velocity1: 500, position2: 3000, velocity2: 100}"

void positionCb(const ethercat_interface::Cmd::ConstPtr & cmd_pos)
{

	uint32 buf32;
	uint16 buf16;
	uint8 buf8;
	bool target1_reached = false;
	bool target2_reached = false;
	/**
	 *Drive state machine transistions
	 *0 -> 6 -> 7 -> 15
	 */
	startup_step1 = 1;
        startup_step2 = 1;
	while (!cmd_executed)
	{	
		///******Motor 2 ***********///
		uint16 statusWord2 = val2->statusWord;	//0x6041
		switch (startup_step2)
		{
			case 1:
				if (homing2)
				{
					WRITE(2, 0x6060, 0, buf8, 6, "OpMode");
				}
				else
				{
					WRITE(2, 0x6060, 0, buf8, 1, "OpMode");
				}
				//sleep(1);
				MotorOn(2);
				startup_step2 = 2;
				break;

			case 2:
				if (homing2)
				{
					target2->controlWord = 0x1f;
					sleep(1);
					if ((val2->statusWord & 0x1400))
					{
						printf("Homing attained for Motor 2\n");
						startup_step2 = 1;
						NewSetPoint(2, false);
						homing2 = false;
					}
					usleep(1000);
				}
				else
				{
					target2->TargetPosition = cmd_pos->position2;
					target2->TargetVelocity = cmd_pos->velocity2;
					NewSetPoint(2, true);
					target2->controlWord = 0x1f;	//5f is relative and 1f is absolute 				
					sleep(1);
					if(TargetReached(2))	//wait till target reached          
					{          
						printf("Target reached for Motor 2\n");
						NewSetPoint(2, false);
						MotorOff(2);
						target2_reached = true;
						usleep(1000);
					}
				}
				break;

			default:
				startup_step2 = 1;
				target2->controlWord = 0x06;	//0x6040
				break;
		}

		uint16 statusWord1 = val1->statusWord;	//0x6041
		switch (startup_step1)
		{
			case 1:
				if (homing1)
				{
					WRITE(1, 0x6060, 0, buf8, 6, "OpMode");
				}
				else
				{
					WRITE(1, 0x6060, 0, buf8, 1, "OpMode");
				}
				//sleep(1);
				MotorOn(1);
				startup_step1 = 2;
				break;

			case 2:
				if (homing1)
				{
					target1->controlWord = 0x1f;
					sleep(1);
					if ((val1->statusWord & 0x1400))
					{
						printf("Homing attained for Motor 1\n");
						homing1 = false;
						startup_step1 = 1;
						NewSetPoint(1, false);
						//target1->controlWord = 0x06;	//0x6040
					}
					usleep(1000);
				}
				else
				{
					target1->TargetPosition = -(cmd_pos->position1);
					target1->TargetVelocity = cmd_pos->velocity1;
					NewSetPoint(1, true);
					target1->controlWord = 0x1f;	//5f is relative and 1f is absolute 				
					sleep(1);
					if(TargetReached(1))	//wait till target reached          
					{          
						printf("Target reached for Motor 1\n");
						NewSetPoint(1, false);
						MotorOff(1);
						target1_reached = true;
						usleep(1000);
					}
				}
				break;

			default:
				startup_step1 = 1;
				target1->controlWord = 0x06;	//0x6040
				break;
		}

		
		printf("Status1: %d Actpos1: %d DI1: %08x Status2: %d Actpos2: %d DI2: %08x ", val1->statusWord, val1->ActPosition, val1->DigitalInputs, val2->statusWord, val2->ActPosition, val2->DigitalInputs);
		printf("TargetPos1: %d, TargetPos2: %d  ", target1->TargetPosition, target2->TargetPosition);

		printf("\n");
		printf("Motor 1: \n");
		printPDSStatus(1);
		printf("Motor 2: \n");
		printPDSStatus(2);
		if((!homing1 && !homing2) && (target1_reached && target2_reached))
		{
			cmd_executed = true;
		}
		//sleep(1);
	}
	cmd_executed = false;

}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "ethercat_interface");

	ros::AsyncSpinner spinner(1);
	spinner.start();

	int freq = 10;	// in Hz

	ros::NodeHandle nh;
	nh.param<int> ("freq", freq, freq);
	ros::Rate r(freq);

	ros::Subscriber pos_cmd_sub = nh.subscribe("/pos_cmd", 1, positionCb);

	std::string ethercat_interface = "enp65s0";

	pthread_create(&thread_statecheck, NULL, ecat_statecheck, (void*) &ctime);
	pthread_create(&thread_pdo, NULL, ecat_pdotransfer, (void*) &ctime);

	// Try to set realtime prio on PDO-thread
	set_realtime_priority(&thread_pdo);

	/*start cyclic part */
	char *interface = new char[ethercat_interface.size() + 1];
	std::copy(ethercat_interface.begin(), ethercat_interface.end(), interface);
	interface[ethercat_interface.size()] = '\0';

	if (start_soem(interface))
	{

		while (ros::ok())
		{

			//To do: add function which publishs driver data here 
			ros::spinOnce();

			r.sleep();
		}
	}

	ROS_INFO("stop transferring messages");
	pdo_transfer_active = FALSE;

	ROS_INFO("stop ethercat");
	stop_ethercat();

	spinner.stop();

	ROS_INFO("Shutdown completed");

	return 0;
}
