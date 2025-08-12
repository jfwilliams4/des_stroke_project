import simpy
import random
import pandas as pd
import simpy.resources
import numpy as np
import matplotlib.pyplot as plt
import csv

# Global class to store parameters for the model.

class g:
    #525600 (Year of Minutes)
    sim_duration = 525600
    number_of_runs = 5
    warm_up_period = sim_duration / 5
    patient_inter = 180
    number_of_nurses = 2
    mean_n_consult_time = 120
    mean_n_ct_time = 20
    number_of_ctp = 1
    sdec_beds = 5
    mean_n_sdec_time = 240
    number_of_ward_beds = 49
    
    # Different variables for ward stay based on diagnosis, thrombolysis and MRS    
    mean_n_i_ward_time_mrs_0 = 1440 * 2
    mean_n_i_ward_time_mrs_1 = 1440 * 3 
    mean_n_i_ward_time_mrs_2 = 1440 * 7 
    mean_n_i_ward_time_mrs_3 = 14400
    mean_n_i_ward_time_mrs_4 = 14400 
    mean_n_i_ward_time_mrs_5 = 14400 * 2 

    mean_n_ich_ward_time_mrs_0 = 17280 
    mean_n_ich_ward_time_mrs_1 = 17280 
    mean_n_ich_ward_time_mrs_2 = 17280 
    mean_n_ich_ward_time_mrs_3 = 17280
    mean_n_ich_ward_time_mrs_4 = 17280 
    mean_n_ich_ward_time_mrs_5 = 17280

    mean_n_non_stroke_ward_time = 4320
    mean_n_tia_ward_time = 1440
    thrombolysis_los_save = 0.75
    
    sdec_dr_cost_min = 0.50
    inpatient_bed_cost = 876
    mean_mrs = 2
    
    # Diagnosis % range
    ich = 10
    i = 60
    tia = 70
    stroke_mimic = 80

    # Admission Range (% Chance of Admission) for TIA and Stroke Mimic, non 
    # stroke shares the range with stroke mimic in this model. (This is 
    # reflected in our real data mainly because most non strokes are often
    #  mimics that are not classified under the stroke mimic criteria in our
    #  data collection)
    tia_admission = 10
    stroke_mimic_admission = 30
    
    # Operational hours of SDEC and CTP are set by the user and stored in the 
    # variables below.

    sdec_unav_time = 0
    sdec_unav_freq = 0 
    ctp_unav_time = 0 
    ctp_unav_freq = 0 

    # These values are changed by the model itself

    sdec_unav = False
    ctp_unav = False
    write_to_csv = False
    gen_graph = False
    therapy_sdec = False
    trials_run_counter = 1

# Patient class to store patient attributes

class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0
        self.q_time_ward = 0
        #0 = known onset, 1 = unknown onset (in ctp range), 2 = unknown (out of
        # ctp range)
        self.onset_type = random.randint(0, 2)
        #Max MRS is set to 5
        self.mrs_type = min(round(random.expovariate(1.0 / g.mean_mrs)), 5)
        self.mrs_discharge = 0
        #<=5 is ICH, <=55 is I, <= 70 is TIA, <=85 is Stroke Mimic, >85 is non\
        # stroke, this set in g class
        self.diagnosis = random.randint(0, 100)
        #0 = ICH, 1 = I, 2 = TIA, 3 = Stroke Mimic, 4 = non stroke
        self.patient_diagnosis = 0
        self.priority = 1
        self.non_admission = random.randint(0, 100)
        self.advanced_ct_pathway = False
        self.sdec_pathway = False
        self.thrombolysis = False
        self.thrombectomy = False
        self.admission_avoidance = False

# Class representing the model of the stroke assessment / treatment process
class Model:
    # Constructor to set up the model for a run. We pass in a run number when
    # we create a new model.
    def __init__(self, run_number):
 
        # Create a SimPy environment
        self.env = simpy.Environment()

        # Create a patient counter for the patient ID
        self.patient_counter = 0

        # Create a SimPy resources to represent stroke nurses, ctp scanners,
        # sdec beds, and ward beds. Set in class g
        self.nurse = simpy.Resource(self.env, 
                                              capacity=g.number_of_nurses)

        self.ctp_scanner = simpy.PriorityResource(self.env,
                                              capacity=g.number_of_ctp)
        
        self.sdec_bed = simpy.PriorityResource(self.env, capacity=g.sdec_beds)

        self.ward_bed = simpy.Resource(self.env, capacity=g.number_of_ward_beds)

        # Store the passed in run number
        self.run_number = run_number

        # Create a Pandas DataFrame that will store a majority of the results 
        # with the patient ID as the index.
        self.results_df = pd.DataFrame()
        self.results_df["Patient ID"] = [1]
        self.results_df["Q Time Nurse"] = [0.0]
        self.results_df["Time with Nurse"] = [0.0]
        self.results_df["Q Time Ward"] = [0.0]
        self.results_df["Ward LOS"] = [0.0]
        self.results_df["Time with CTP"] = [0.0]
        self.results_df["Time with CT"] = [0.0]
        self.results_df["Time in SDEC"] = [0.0]  
        self.results_df["CTP Status"] = [""]
        self.results_df["SDEC Status"] = [""]
        self.results_df["Thrombolysis"] = [""]
        self.results_df["SDEC Occupancy"] = [0.0]
        self.results_df["Admission Avoidance"] = [""]
        self.results_df["MRS Type"] = [0.0]
        self.results_df["MRS DC"] = [0.0]
        self.results_df["MRS Change"] = [0.0]
        self.results_df["Onset Type"] = [0.0]
        self.results_df["Diagnosis Type"] = [""]
        self.results_df["Thrombolysis Savings"] = [0.0]
        self.results_df["Ward Occupancy"] = [0.0]
        self.results_df.set_index("Patient ID", inplace=True)

        # A variable to count the number of SDEC freezes
        self.sdec_freeze_counter = 0

        # Create a variable to store the mean queuing time for the nurse
        self.mean_q_time_nurse = 0

        # Create a variable to store the mean time waiting for a ward bed
        self.mean_q_time_ward = 0

        # Create a variable to store the mean length of stay in the ward
        self.mean_los_ward = 0

        # Create a variable to store the mean number of thrombolysis savings
        self.thrombolysis_savings = 0

        # set up a list to store the queue for stroke nurse assessment
        self.q_for_assessment = []

        # a PD dataframe for the assessment queue graph
        self.nurse_q_graph_df = pd.DataFrame()
        self.nurse_q_graph_df["Time"] = [0.0]
        self.nurse_q_graph_df["Patients in Assessment Queue"] = [0.0]

        # a list that will store the number of patients in the SDEC
        self.sdec_occupancy = []

        # A list that will store the number of admissions avoided
        self.admission_avoidance = []

        # A list that will store the number of patients in the ward
        self.ward_occupancy = []
        
        # A list to store the number of patients avoiding admission
        self.non_admissions = []

        self.occupancy_graph_df = pd.DataFrame()
        self.occupancy_graph_df["Time"] = [0.0]
        self.occupancy_graph_df["Ward Occupancy"] = [0.0]

    # A generator function for the patient arrivals. This is an infinite loop
    def generator_patient_arrivals(self):

        while True:
            # Increment the patient counter by 1 for each new patient
            self.patient_counter += 1
            
            # Create a new patient - an instance of the Patient Class we
            # defined above. patient counter ID passed from above to patient 
            # class.
            p = Patient(self.patient_counter)

            # Tell SimPy to start the stroke assessment function with
            # this patient (the generator function that will model the
            # patient's journey through the system)
            self.env.process(self.stroke_assessment(p))

            # Randomly sample the time to the next patient arriving.
            sampled_inter = random.expovariate(1.0 / g.patient_inter)

            # Freeze this instance of this function in place until the
            # inter-arrival time has elapsed.
            yield self.env.timeout(sampled_inter)

    def obstruct_ctp(self):
        while True:
            yield self.env.timeout(g.ctp_unav_freq)
            # Once elapsed, this generator requests the ctp scanner with
            # a priority of -1. As the patient priority is set at 1
            # the scanner will take priority over any patients waiting. 
            # This method also means that the scanner won't stop mid scan. 
            g.ctp_unav = True    
            with self.ctp_scanner.request(priority=-1) as req:
                yield req
                    
                # Freeze with the scanners held in place for the unavailability 
                # time, in the model this means patients admitted in this time 
                # will not have a ctp scan.  
                # freq and unav times are set in the g class
                yield self.env.timeout(g.ctp_unav_time)
                g.ctp_unav = False
    
    def obstruct_sdec(self):
        while True:
            yield self.env.timeout(g.sdec_unav_freq)
            # Once elapsed, this generator requests the SDEC with
            # a priority of -1. As the patient priority is set at 1
            # the SDEC will take priority over any patients waiting.  
            g.sdec_unav = True  
            with self.sdec_bed.request(priority=-1) as req:
                yield req
                    
                # Freeze with the SDEC held in place for the unavailability 
                # time, in the model this means patients admitted in this time 
                # will not have passed through the SDEC.  
                # freq and unav times are set in the g class
                yield self.env.timeout(g.sdec_unav_time)
                g.sdec_unav = False
                if self.env.now > g.warm_up_period:
                    self.sdec_freeze_counter += 1
    
    # A generator function that represents the pathway for a patient going
    # through the stroke assessment process.
    # The patient object is passed in to the generator function so we can 
    # extract information from / record information to it
    def stroke_assessment(self, patient):

        # This code introduces a slight element of randomness into the patient's
        # diagnosis.

        self.ich_range = random.normalvariate(g.ich, 1)
        self.i_range = max(random.normalvariate(g.i, 1), self.ich_range)
        self.tia_range = max(random.normalvariate(g.tia, 1), self.i_range)
        self.stroke_mimic_range = max(random.normalvariate(g.stroke_mimic, 1), 
                                      self.tia_range)
        self.non_stroke_range = max(random.normalvariate(g.stroke_mimic, 1), 
                                    self.stroke_mimic_range)
        
        if patient.diagnosis <= self.ich_range:
            patient.patient_diagnosis = 0
        elif patient.diagnosis <= self.i_range:
            patient.patient_diagnosis = 1
        elif patient.diagnosis <= self.tia_range:
            patient.patient_diagnosis = 2
        elif patient.diagnosis <= self.stroke_mimic_range:
            patient.patient_diagnosis = 3
        elif patient.diagnosis > self.non_stroke_range:
            patient.patient_diagnosis = 4

        # Record the time the patient started queuing for a nurse
        start_q_nurse = self.env.now

        self.q_for_assessment.append(patient)

        # This code says request a nurse resource, and do all of the following
        # block of code with that nurse resource held in place (and therefore
        # not usable by another patient)
        with self.nurse.request() as req:
            # Freeze the function until the request for a nurse can be met.
            # The patient is currently queuing.
            yield req

            # Control is passed back to the generator function once the request
            # is met for a nurse. As the queue for the nurse is finished 
            # the patient then leaves the assessment queue list.
            
            end_q_nurse = self.env.now

            self.q_for_assessment.remove(patient)

            # The code below checks if the warm up period has passed before 
            # entering data into the df, this code exists when ever data is 
            # recorded  

            if self.env.now > g.warm_up_period:
                self.nurse_q_graph_df.loc[len(self.nurse_q_graph_df)] = [
                    self.env.now,
                    len(self.q_for_assessment)]

            # Calculate the time this patient was queuing for the nurse, and
            # record it in the patient's attribute
            patient.q_time_nurse = end_q_nurse - start_q_nurse

            # The below code creates a random action time for the nurse based 
            # on the mean in g class, and assigns it ot a variable. Currently 
            # using a Exponential distribution but might need to switch to 
            # a Log normal one (though the intense variation in the real life
            # consult time might mean a exponetial distribution is better)
            sampled_nurse_act_time = random.expovariate(1.0 / 
                                                        g.mean_n_consult_time)
                
            # Freeze this function in place for the activity time we sampled
            # above.  This is the patient spending time with the nurse.
            yield self.env.timeout(sampled_nurse_act_time)

            # In the .at function below, the first value is the row, the second
            # value is the column in which to add data. The final value is the 
            # the data that is to be added to the DF, in this case the Nurse 
            # Q time

            if self.env.now > g.warm_up_period:
                self.results_df.at[patient.id, "Q Time Nurse"] = (
                    patient.q_time_nurse)
                self.results_df.at[patient.id, "Time with Nurse"] = (
                    sampled_nurse_act_time)

        # The if formula below checks to see if the CTP scanner is active 
        # and if it is the following code is followed including updating the 
        # patient advanced CT pathway attribute
        
        if g.ctp_unav == False:
        
            patient.advanced_ct_pathway = True
        
            # Randomly sample the mean ct time, as with above this may need to 
            # be updated to a log normal distribution 

            sampled_ctp_act_time = random.expovariate(1.0 / 
                                                        g.mean_n_ct_time)
            
            # Freeze this function in place for the activity time that was 
            # sampled above.
            yield self.env.timeout(sampled_ctp_act_time)

            # Add data to the DF afer the warm up period.

            if self.env.now > g.warm_up_period:
                self.results_df.at[patient.id, "Time with CTP"] = (
                sampled_ctp_act_time)

        # If the CTP pathway is not active the below code runs, it is the same 
        # as the above however adds data to a different column and the patient 
        # advanced CT pathway remains False.

        else:

            sampled_ct_act_time = random.expovariate(1.0 / 
                                                        g.mean_n_ct_time)
                
            yield self.env.timeout(sampled_ct_act_time)    


            if self.env.now > g.warm_up_period:
                self.results_df.at[patient.id, "Time with CT"] = (
                sampled_ct_act_time)   

        # The below code records the status of both the CTP and SDEC pathways.
        # Both exist as generators and this data is record to ensure they are 
        # operating as expected.

        if self.env.now > g.warm_up_period: 
            self.results_df.at[patient.id, "CTP Status"] = (
            g.ctp_unav)

        if self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "SDEC Status"] = (
            g.sdec_unav)

        # The if statement below checks if the SDEC pathway is active at this 
        # given time and if there is space in the SDEC itself.

        if g.sdec_unav == False and len(self.sdec_occupancy) <= g.sdec_beds:
                
            # If the conditions above are met the patient attribute for the SDEC
            # are changed to True and the patient is added to the SDEC occupancy
            # list.

            self.sdec_occupancy.append(patient)

            # The below code record the SDEC Occupancy as the patient passes 
            # this point to ensure it is working as expected.

            if self.env.now > g.warm_up_period:
                self.results_df.at[patient.id, "SDEC Occupancy"] = (
                len(self.sdec_occupancy))

            patient.sdec_pathway = True
            

            sampled_sdec_stay_time = random.expovariate(1.0 / 
                                                        g.mean_n_sdec_time)
            
            # Freeze this function in place for the activity time we sampled
            # above.
            yield self.env.timeout(sampled_sdec_stay_time)
            
            # This code checks if the ward is full, if this is the case the 
            # patient will not be released from the SDEC, thus impeding it use  

            while len(self.ward_occupancy) >= g.number_of_ward_beds:
                    yield self.env.timeout(1)

            # Once the above code is complete the patient is removed from the 
            # SDEC occupancy list.

            self.sdec_occupancy.remove(patient)

            # Code to record the SDEC stay time in the results DataFrame.

            if self.env.now > g.warm_up_period:
                self.results_df.at[patient.id, "Time in SDEC"] =\
                      (sampled_sdec_stay_time)

        # The below code checks the patient's attributes to see if the 
        # thrombolysis attribute should be changed to True, this is based off 
        # the patient diagnosis, onset type and mrs type. There are different 
        # conditions depending on if CTP is available or not.

        if patient.patient_diagnosis == 1 and patient.onset_type == 0 \
            and patient.mrs_type > 0:
            patient.thrombolysis = True

        if patient.patient_diagnosis == 1 and patient.onset_type == 1 and\
              patient.advanced_ct_pathway == True and patient.mrs_type > 0:
            patient.thrombolysis = True

        # Thrombolysis status is added to the DF, this is mainly used to check 
        # if it is being applied correctly.

        if self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "Thrombolysis"] = (
                patient.thrombolysis)

        # The below code records the patients diagnosis attribute, this is added
        # to the DF to check the diagnosis code is working correctly.

        if patient.patient_diagnosis == 0 and self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "Diagnosis Type"] = "ICH"
        elif patient.patient_diagnosis == 1 and self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "Diagnosis Type"] = "I"
        elif patient.patient_diagnosis == 2 and self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "Diagnosis Type"] = "TIA"
        elif patient.patient_diagnosis == 3 and self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "Diagnosis Type"] = "Stroke Mimic"
        elif patient.patient_diagnosis == 4 and self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "Diagnosis Type"] = "Non Stroke"
        
        if self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "Onset Type"] = (
            patient.onset_type)

        # This code checks if the patient is eligible for admission avoidance 
        # with no therapy support enabled

        if g.therapy_sdec == False:  
            
            if patient.patient_diagnosis < 2 and patient.sdec_pathway == \
                True and patient.mrs_type < 2 and patient.thrombolysis == False:
                
                patient.admission_avoidance = True
                if self.env.now > g.warm_up_period:
                    self.results_df.at[patient.id, "Admission Avoidance"] = (
                    patient.sdec_pathway)

        # This code checks if the patient is eligible for admission avoidance 
        # with therapy support enabled

        if g.therapy_sdec == True:
            
            if patient.patient_diagnosis < 2 and patient.sdec_pathway == \
                True and patient.mrs_type < 3 and patient.thrombolysis == False:
                
                patient.admission_avoidance = True
                if self.env.now > g.warm_up_period:
                    self.results_df.at[patient.id, "Admission Avoidance"] = (
                    patient.sdec_pathway)

        # This code adds the Patient's MRS to the DF, this can be used to check
        # all code that interacts with this runs correctly.

        if self.env.now > g.warm_up_period:
            self.results_df.at[patient.id, "MRS Type"] = (
            patient.mrs_type)
        
        # Patients with a True admission avoidance are added to a list that is 
        # used to calculate the savings from the avoided admissions. 

        if patient.admission_avoidance == True and self.env.now > \
            g.warm_up_period:
            self.admission_avoidance.append(patient)

        # This code introduces a small element of randomness into the admission
        # rates for the non stroke, tia and stroke mimic patients.

        self.tia_admission_chance = random.normalvariate(g.tia_admission, 1)
        self.stroke_mimic_admission_chance = random.normalvariate(
            g.stroke_mimic_admission, 1)
        
        # This code exists after the admission avoidance code so they are not 
        # added to the admission avoidance list, as that should only be for 
        # SDEC patients who avoid admission. This code checks if TIA, non stroke
        # and stroke mimic patients should be admitted based on the values 
        # established in the previous code and g class. (To me it looks like 
        # the < & > signs are the wrong way round, but this is correct).

        if patient.non_admission >= self.tia_admission_chance and \
            patient.patient_diagnosis == 2:
            patient.admission_avoidance = True

        if patient.non_admission >= self.stroke_mimic_admission_chance and \
            patient.patient_diagnosis > 2:
            patient.admission_avoidance = True            

        # once all the above code has been run all patients who will not admit
        # have a True admission avoidance attribute. For all the patients that 
        # remain false, the below code will run simulating the admission to the 
        # ward.

        if patient.admission_avoidance != True:

            # These code assigns a time to the start q variable. In stroke care
            # delays can have serious consequence so modeling this is very
            # important as flow disruption are a common issue. 

            start_q_ward = self.env.now

            # Request the ward bed and hold the patient in a queue until this 
            # is met.

            with self.ward_bed.request() as req:

                yield req

                # Add patient to the ward list

                self.ward_occupancy.append(patient)

                if self.env.now > g.warm_up_period:
                    self.results_df.at[patient.id, "Ward Occupancy"] = (
                    len(self.ward_occupancy))

                if self.env.now > g.warm_up_period:
                    self.occupancy_graph_df.loc[len(self.occupancy_graph_df)] =\
                        [self.env.now,
                    len(self.ward_occupancy)]

                # The patient attribute for the queuing time in the ward is 
                # assigned here.

                end_q_ward = self.env.now

                patient.q_time_ward = end_q_ward - start_q_ward

                # The below code checks the patients diagnosis and MRS,
                # adjusting MRS change and LOS baised on these. This code is 
                # for ICH patients. 

                if patient.patient_diagnosis == 0 and patient.mrs_type == 0:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_ich_ward_time_mrs_0)
                    patient.mrs_discharge = patient.mrs_type
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 0 and patient.mrs_type == 1:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_ich_ward_time_mrs_1)
                    patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 0 and patient.mrs_type == 2:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_ich_ward_time_mrs_2)
                    patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 0 and patient.mrs_type == 3:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_ich_ward_time_mrs_3)
                    patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 0 and patient.mrs_type == 4:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_ich_ward_time_mrs_4)
                    patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 0 and patient.mrs_type == 5:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_ich_ward_time_mrs_5)
                    patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)
                
                # The below code checks the patients diagnosis and MRS,
                # adjusting MRS change and LOS baised on these. This code is 
                # for I patients amd also checks for thrombolysis and adjusts 
                # LOS and associated savings accordingly. 
                
                if patient.patient_diagnosis == 1 and patient.mrs_type == 0:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_i_ward_time_mrs_0)
                    patient.mrs_discharge = patient.mrs_type
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 1 and patient.mrs_type == 1:
                    sampled_ward_act_time = random.expovariate\
                    (1.0 / g.mean_n_i_ward_time_mrs_1)
                    if patient.thrombolysis == True:
                        sampled_ward_act_time_thrombolysis = \
                                sampled_ward_act_time * g.thrombolysis_los_save
                        patient.mrs_discharge = patient.mrs_type - \
                            random.randint(0,1)
                        yield self.env.timeout(\
                            sampled_ward_act_time_thrombolysis)
                        if self.env.now > g.warm_up_period and\
                              patient.advanced_ct_pathway == True:
                            self.results_df.at[patient.id,\
                         "Thrombolysis Savings"] = (((sampled_ward_act_time\
                         - sampled_ward_act_time_thrombolysis)/60)/24)*\
                            g.inpatient_bed_cost
                        self.ward_occupancy.remove(patient)
                    else:
                        patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                        yield self.env.timeout(sampled_ward_act_time)
                        self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 1 and patient.mrs_type == 2:
                    sampled_ward_act_time = random.expovariate\
                    (1.0 / g.mean_n_i_ward_time_mrs_2)
                    if patient.thrombolysis == True:
                        sampled_ward_act_time_thrombolysis = \
                                sampled_ward_act_time * g.thrombolysis_los_save
                        patient.mrs_discharge = patient.mrs_type - \
                            random.randint(0,2)
                        yield self.env.timeout(\
                            sampled_ward_act_time_thrombolysis)
                        if self.env.now > g.warm_up_period and\
                              patient.advanced_ct_pathway == True:
                            self.results_df.at[patient.id,\
                         "Thrombolysis Savings"] = (((sampled_ward_act_time\
                         - sampled_ward_act_time_thrombolysis)/60)/24)*\
                            g.inpatient_bed_cost
                        self.ward_occupancy.remove(patient)
                    else:
                        patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                        yield self.env.timeout(sampled_ward_act_time)
                        self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 1 and patient.mrs_type == 3:
                    sampled_ward_act_time = random.expovariate\
                    (1.0 / g.mean_n_i_ward_time_mrs_3)
                    if patient.thrombolysis == True:
                        sampled_ward_act_time_thrombolysis = \
                                sampled_ward_act_time * g.thrombolysis_los_save
                        patient.mrs_discharge = patient.mrs_type - \
                            random.randint(0,2)
                        yield self.env.timeout(\
                            sampled_ward_act_time_thrombolysis)
                        if self.env.now > g.warm_up_period and\
                              patient.advanced_ct_pathway == True:
                            self.results_df.at[patient.id,\
                         "Thrombolysis Savings"] = (((sampled_ward_act_time\
                         - sampled_ward_act_time_thrombolysis)/60)/24)*\
                            g.inpatient_bed_cost
                        self.ward_occupancy.remove(patient)
                    else:
                        patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                        yield self.env.timeout(sampled_ward_act_time)
                        self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 1 and patient.mrs_type == 4:
                    sampled_ward_act_time = random.expovariate\
                    (1.0 / g.mean_n_i_ward_time_mrs_4)
                    if patient.thrombolysis == True:
                        sampled_ward_act_time_thrombolysis = \
                                sampled_ward_act_time * g.thrombolysis_los_save
                        patient.mrs_discharge = patient.mrs_type - \
                            random.randint(0,2)
                        yield self.env.timeout(\
                            sampled_ward_act_time_thrombolysis)
                        if self.env.now > g.warm_up_period and\
                              patient.advanced_ct_pathway == True:
                            self.results_df.at[patient.id,\
                         "Thrombolysis Savings"] = (((sampled_ward_act_time\
                         - sampled_ward_act_time_thrombolysis)/60)/24)*\
                            g.inpatient_bed_cost
                        self.ward_occupancy.remove(patient)
                    else:
                        patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                        yield self.env.timeout(sampled_ward_act_time)
                        self.ward_occupancy.remove(patient)

                elif patient.patient_diagnosis == 1 and patient.mrs_type == 5:
                    sampled_ward_act_time = random.expovariate\
                    (1.0 / g.mean_n_i_ward_time_mrs_5)
                    if patient.thrombolysis == True:
                        sampled_ward_act_time_thrombolysis = \
                                sampled_ward_act_time * g.thrombolysis_los_save
                        patient.mrs_discharge = patient.mrs_type - \
                            random.randint(0,2)
                        yield self.env.timeout(\
                            sampled_ward_act_time_thrombolysis)
                        if self.env.now > g.warm_up_period and\
                              patient.advanced_ct_pathway == True:
                            self.results_df.at[patient.id,\
                         "Thrombolysis Savings"] = (((sampled_ward_act_time\
                         - sampled_ward_act_time_thrombolysis)/60)/24)*\
                            g.inpatient_bed_cost
                        self.ward_occupancy.remove(patient)
                    else:
                        patient.mrs_discharge = patient.mrs_type -\
                              random.randint(0,1)
                        yield self.env.timeout(sampled_ward_act_time)
                        self.ward_occupancy.remove(patient)

            # The below code is for the non stroke diagnosis.

                if patient.patient_diagnosis == 2:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_tia_ward_time)
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)
                
                if patient.patient_diagnosis > 2:
                    sampled_ward_act_time = random.expovariate\
                        (1.0 / g.mean_n_non_stroke_ward_time)
                    yield self.env.timeout(sampled_ward_act_time)
                    self.ward_occupancy.remove(patient)
            
            # Relevent information is recorded in the results DataFrame.

            if self.env.now > g.warm_up_period:
                    self.results_df.at[patient.id, "Q Time Ward"] = (
                    patient.q_time_ward)
                    self.results_df.at[patient.id, "Ward LOS"] = (
                    sampled_ward_act_time)
                    self.results_df.at[patient.id, "MRS DC"] = (
                    patient.mrs_discharge)
                    self.results_df.at[patient.id, "MRS Change"] = (
                    patient.mrs_type - patient.mrs_discharge)


    # This method calculates results over a single run.
    
    def calculate_run_results(self):
       
        # Drop the first row of the results DataFrame, as this is just a dummy
        # and will take on the value of zero.
        self.results_df.drop([1], inplace=True)

        # The below code calculates the average or cumulative values the model 
        # is concerned with.

        self.mean_q_time_nurse = round(self.results_df["Q Time Nurse"].mean(),0)

        self.number_of_admissions_avoided = len(self.admission_avoidance)

        self.mean_q_time_ward = round(self.results_df["Q Time Ward"].mean()/\
                                      60, 0)

        self.mean_ward_occupancy = round(self.results_df["Ward Occupancy"].\
                                         mean())

        self.admission_delays = len(self.results_df\
                                    [self.results_df["Q Time Ward"] > 0])

        self.mean_los_ward = round(self.results_df["Ward LOS"].mean()/60, 0)

        self.sdec_financial_savings = len(self.admission_avoidance) * \
            g.inpatient_bed_cost
        
        # The below code ensures that the SDEC incurs no cost if it is not 
        # running at all in the model. This was introduced as a bug was causing
        # it to return small values even if the SDEC was not running. This is 
        # now fixed, but the code works so I have left it in place.

        if g.sdec_unav_freq == 0:
            self.medical_staff_cost = 0
        else:
            self.medical_staff_cost = round(
            g.sdec_dr_cost_min * (g.sim_duration ) - \
            g.sdec_dr_cost_min * self.sdec_freeze_counter * g.sdec_unav_time, 0)
        
        self.savings_sdec = round(self.sdec_financial_savings - \
            self.medical_staff_cost, 0)
        
        self.thrombolysis_savings = round(self.results_df\
                                          ["Thrombolysis Savings"].sum(),0)
        self.total_savings = self.thrombolysis_savings + self.savings_sdec

        self.mean_mrs_change = round(self.results_df["MRS Change"].mean(), 2)

    # This method plots the stroke nurse assessment queue graph, as it is after
    # the run method it will appear after the run has completed in the output.
    # Might need to change this...

    def plot_stroke_run_graphs(self):  
        
        if g.gen_graph == True:
        
            # Queue for Nurse Assessment Graph (Currently Commented Out)

            #self.nurse_q_graph_df.drop([0], inplace=True)
                
            #fig, ax = plt.subplots()

            #ax.set_xlabel("Time")
            #ax.set_ylabel("Number of patients in Q for Assessment")
            #ax.set_title(f"Number of Patients in Nurse Assessment Queue \
                         #Over Time "f"{self.run_number}")

            #ax.plot(self.nurse_q_graph_df["Time"],
                    #self.nurse_q_graph_df["Patients in Assessment Queue"],
                    #color="m",
                    #linestyle="-",
                    #label="Q for Stroke Nurse Assessment")
            
            #ax.legend(loc="upper right")
            
            #fig.show()

            # Ward Occupancy Graph

            self.occupancy_graph_df.drop([0], inplace=True)

            fig, ax = plt.subplots()

            ax.set_xlabel("Time")
            ax.set_ylabel("Stroke Ward Occupancy")
            ax.set_title(f"Trial "f"{g.trials_run_counter}\
                         Ward Occupancy Over Time "f"{self.run_number}")

            ax.plot(self.occupancy_graph_df["Time"],
                    self.occupancy_graph_df["Ward Occupancy"],
                    color="b",
                    linestyle="-",
                    label="Ward Occupancy")
            
            # Add trend line 
            x = self.occupancy_graph_df["Time"]
            y = self.occupancy_graph_df["Ward Occupancy"]
            z = np.polyfit(x, y, 1)  # 1 = linear fit
            p = np.poly1d(z)
            ax.plot(x, p(x), color="b", linestyle="--", label="Trend Line")
 
            ax.legend(loc="upper right")
            
            fig.show()
            
        
    # The run method starts up the DES entity generators, runs the simulation,
    # and in turns calls anything we need to generate results for the run
    
    def run(self):
        
        # starts up the generators in the model, of which there are three.

        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.obstruct_ctp())
        self.env.process(self.obstruct_sdec())

        # Run the model for the duration specified in g class
        self.env.run(until=(g.sim_duration + g.warm_up_period))

        # Now the simulation run has finished, call the method that calculates
        # run results
        self.calculate_run_results()

        # Print the run number with the patient-level results from this run of 
        # the model, this is commented out at the moment. 

        #print (f"Run Number {self.run_number}")
        #print (self.results_df)

        if g.write_to_csv == True:
            self.results_df.to_csv\
                (f"trial {g.trials_run_counter} output {self.run_number}.csv", 
                               index=False)

        self.plot_stroke_run_graphs()
    
# Class representing a Trial for our simulation - a batch of simulation runs.

class Trial:
    
    # The constructor sets up a pandas dataframe that will store the key
    # results from each run with run number as the index.
    
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Nurse (Mins)"] = [0.0]
        self.df_trial_results["Number of Admissions Avoided In Run"] = [0.0]
        self.df_trial_results["Mean Q Time Ward (Hour)"] = [0.0]
        self.df_trial_results["Mean Occupancy"] = [0.0]
        self.df_trial_results["Number of Admission Delays"] = [0.0]
        self.df_trial_results["Mean Length of Stay Ward (Hours)"] = [0.0]
        self.df_trial_results["Financial Savings of Admissions Avoidance (£)"]=\
              [0.0]
        self.df_trial_results["SDEC Medical Staff Cost (£)"] = [0.0]
        self.df_trial_results["SDEC Savings (£)"] = [0.0]
        self.df_trial_results["Thrombolysis Savings (£)"] = [0.0]
        self.df_trial_results["Total Savings"] = [0.0]
        self.df_trial_results["Mean MRS Change"] = [0.0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to run a trial
    
    def run_trial(self):

        # Run the simulation for the number of runs specified in g class.
        # For each run, we create a new instance of the Model class and call its
        # run method, which sets everything else in motion.  Once the run has
        # completed, we grab out the stored run results 
        # and store it against the run number in the trial results dataframe.
        
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run()
            
            self.df_trial_results.loc[run] = [my_model.mean_q_time_nurse,
                                              my_model.\
                                                number_of_admissions_avoided,
                                                my_model.mean_q_time_ward,
                                                my_model.mean_ward_occupancy,
                                                my_model.admission_delays,
                                                my_model.mean_los_ward,
                                                my_model.sdec_financial_savings,
                                                my_model.medical_staff_cost,
                                                my_model.savings_sdec,
                                                my_model.thrombolysis_savings,
                                                my_model.total_savings,
                                                my_model.mean_mrs_change]

        if g.write_to_csv == True:
            self.df_trial_results.to_csv\
                (f"trial {g.trials_run_counter} trial results.csv", 
                               index=False)

        # This is new code that will store all averages to compare across 
        # the different trials. It does this by checking if the attribute
        # exists in the global g class, and if it doesn't it creates it. It 
        # then stores the mean of each run against the attribute 
        # (eg "trial_mean_q_time_nurse")

        # The mean is stored against the key of g.trials_run_counter.

        for attr, col in [("trial_mean_q_time_nurse",\
                           "Mean Q Time Nurse (Mins)"),
            ("trial_number_of_admissions_avoided",\
              "Number of Admissions Avoided In Run"),
            ("trial_mean_q_time_ward", "Mean Q Time Ward (Hour)"),
            ("trial_mean_occupancy", "Mean Occupancy"),
            ("trial_number_of_admission_delays", "Number of Admission Delays"),
            ("trial_financial_savings_of_a_a",\
             "Financial Savings of Admissions Avoidance (£)"),\
                ("sdec_medical_cost","SDEC Medical Staff Cost (£)"),
            ("trial_sdec_financial_savings", "SDEC Savings (£)"),
            ("trial_thrombolysis_savings", "Thrombolysis Savings (£)"),
            ("trial_total_savings", "Total Savings"),("trial_mrs_change",\
                                                      "Mean MRS Change")]:

        # Checks to see if the attribute already exists and if it doesn't 
        # create it. Creates a mean of each trial and creates a dictionary 
        # that can be read later.

            if not hasattr(g, attr):
                setattr(g, attr, {})
            getattr(g, attr)[g.trials_run_counter] = \
                round(self.df_trial_results[col].mean(), 2)

        # Code to store the configuration that was used for this trial.
        self.trial_info = (
            f"Trial {g.trials_run_counter}, SDEC Therapy = {g.therapy_sdec},"\
                 f" SDEC Open % = {sdec_value}, CTP Open % = {ctp_value}")

        print ("---------------------------------------------------")
        print(f"{self.trial_info}")
        print(f"Trial {g.trials_run_counter} Results:")
        print (" ")
        print(f"Trial Mean Q Time Nurse (Mins):     \
              {g.trial_mean_q_time_nurse[g.trials_run_counter]}")
        print(f"Trial Number of Admissions Avoided: \
              {g.trial_number_of_admissions_avoided[g.trials_run_counter]}")
        print(f"Trial Mean Q Time Ward (Hours):     \
              {g.trial_mean_q_time_ward[g.trials_run_counter]}")
        print(f"Trial Mean Ward Occupancy:          \
              {g.trial_mean_occupancy[g.trials_run_counter]}")
        print(f"Trial Number of Admission Delays:   \
              {g.trial_number_of_admission_delays[g.trials_run_counter]}")
        print(f"Trial SDEC Total Savings (£):       \
              {g.trial_financial_savings_of_a_a[g.trials_run_counter]}")
        print(f"Trial SDEC Medical Cost (£):        \
              {g.sdec_medical_cost[g.trials_run_counter]}")
        print(f"Trial SDEC Savings - Cost (£):      \
              {g.trial_sdec_financial_savings[g.trials_run_counter]}")
        print(f"Trial Thrombolysis Savings (£):     \
              {g.trial_thrombolysis_savings[g.trials_run_counter]}")
        print(f"Trial Total Savings (£):            \
              {g.trial_total_savings[g.trials_run_counter]}")
        print(f"Mean MRS Change:                    \
              {g.trial_mrs_change[g.trials_run_counter]}")
        
#This code asks the user if they want to generate cvs per run
csv_input = False

while csv_input == False:

    csv_value = input ("Write results to CSV? Yes / No")
    if csv_value == "Yes" or csv_value == "yes":
        g.write_to_csv = True
        csv_input = True
    elif csv_value == "No" or csv_value == "no":
        g.write_to_csv = False
        csv_input = True
    else:
        print ("Invalid Input Please Try Again")

#This code asks the user if they want to generate a graph per run
graph_input = False

while graph_input == False:

    graph_value = input ("Generate graph per run? Yes / No")
    if graph_value == "Yes" or graph_value == "yes":
        g.gen_graph = True
        graph_input = True
    elif graph_value == "No" or graph_value == "no":
        g.gen_graph = False
        graph_input = True
    else:
        print ("Invalid Input Please Try Again")


for x in range(3):

    # This code asks if the user wants to have full therapy support for the SDEC

    therapy_input = False

    while therapy_input == False:

        therapy_value = input ("Run SDEC with Full Therapy Support? Yes / No")
        if therapy_value == "Yes" or therapy_value == "yes":
            g.therapy_sdec = True
            therapy_input = True
        elif therapy_value == "No" or therapy_value == "no":
            g.therapy_sdec = False
            therapy_input = True
        else:
            print ("Invalid Input Please Try Again")

    # This code asks the user how long the SDEC should be unavailable for, as a 
    # % of days.

    sdec_input = False

    while sdec_input == False:

        sdec_value = int(input("What percentage of the day should the SDEC " \
        "be available? (0-100)"))
        if sdec_value <= 100 and sdec_value >= 0:
            g.sdec_unav_freq = 1440 * (sdec_value / 100)
            g.sdec_unav_time = 1440 - g.sdec_unav_freq
            sdec_input = True
        elif sdec_value == 100:
            g.sdec_unav_freq = g.sim_duration * 2
            g.sdec_unav_time = 0
            sdec_input = True
        else:
            print ("Invalid Input Please Try Again")

    # This code asks the user how long the SDEC should be unavailable for, as a 
    # % of days.

    ctp_input = False

    while ctp_input == False:

        ctp_value = int(input("What percentage of the day should the CTP " \
        "be available? (0-100)"))
        if ctp_value <= 100 and ctp_value >= 0:
            g.ctp_unav_freq = 1440 * (ctp_value / 100)
            g.ctp_unav_time = 1440 - g.ctp_unav_freq
            ctp_input = True
        elif sdec_value == 100:
            g.ctp_unav_freq = g.sim_duration * 2
            g.ctp_unav_time = 0
            sdec_input = True
        else:
            print ("Invalid Input Please Try Again")

    # Create an instance of the Trial class
    my_trial = Trial()

    # Call the run_trial method of our Trial object
    my_trial.run_trial()

    g.trials_run_counter += 1

print ("All Trials Completed")


# Combine all trial results into a single dictionary, I am 
# currently unaware were the trial_sdec_finacial_savings is stored in class g
# but it works so I'll leave it for now...
trial_numbers = g.trial_sdec_financial_savings.keys()
combined_results = {
    trial: {
        "Mean Q Time Nurse (Mins)": g.trial_mean_q_time_nurse.get(trial, None),
        "Number of Admissions Avoided In Run": \
            g.trial_number_of_admissions_avoided.get(trial, None),
        "Mean Q Time Ward (Hours)": g.trial_mean_q_time_ward.get(trial, None),
        "Mean Occupancy": g.trial_mean_occupancy.get(trial, None),
        "Number of Admission Delays": \
            g.trial_number_of_admission_delays.get(trial, None),
        "Total SDEC Savings (£)":\
            g.trial_financial_savings_of_a_a.get(trial, None),\
            "Total SDEC Staff Cost (£)": g.sdec_medical_cost.get(trial, None),
        "SDEC Savings - Costs (£)": \
            g.trial_sdec_financial_savings.get(trial, None),
        "Thrombolysis Savings (£)":\
              g.trial_thrombolysis_savings.get(trial, None),
        "Total Savings (£)": g.trial_total_savings.get(trial, None),\
            "Mean MRS Change": g.trial_mrs_change.get(trial, None)}
    for trial in trial_numbers
}

df_all_trial_results = pd.DataFrame.from_dict(combined_results, orient='index')
df_all_trial_results.index.name = 'Trial Number'

if g.write_to_csv == True:
    df_all_trial_results.to_csv("all_trial_results.csv", 
                               index=False)
