# Imports the SPEnv library, which will perform the Agent actions themselves
from spEnv import SpEnv

# Callback used to print the results at each episode
from callback import ValidationCallback

import keras
# Keras library for the NN considered
from keras.models import Sequential

# Keras libraries for layers, activations and optimizers used
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam

# RL Agent 
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

# Mathematical operations used later
from math import floor

#Library to manipulate the dataset in a csv file
import pandas as pd

# Library used to manipulate time
import datetime

class DeepQTrading:
    '''
    Class constructor
    model: Keras model considered
    Explorations is a vector containing (i) probability of random predictions; (ii) how many epochs will be 
    runned by the algorithm (we run the algorithm several times-several iterations)  
    trainSize: size of the training set
    validationSize: size of the validation set
    testSize: size of the testing set 
    outputFile: name of the file to print results
    begin: Initial date
    end: final date
    nbActions: number of decisions (0-Hold 1-Long 2-Short) 
    nOutput is the number of walks. We are doing 5 walks.  
    operationCost: Price for the transaction (we set they are free)
    '''
    def __init__(self, ticker, model, explorations, trainSize, validationSize, testSize, outputFile, begin, end, nbActions, isOnlyShort, ensembleFolderName, operationCost=0):
        
        self.isOnlyShort = isOnlyShort
        self.ensembleFolderName = ensembleFolderName

        # Define the policy, explorations, actions, ticker and model as received by parameters
        self.policy = EpsGreedyQPolicy()
        self.explorations = explorations
        self.nbActions = nbActions
        self.model = model
        self.ticker = ticker

        # Define the memory
        self.memory = SequentialMemory(limit=10000, window_length=1)

        # Instantiate the agent with parameters received
        self.agent = DQNAgent(model=self.model, policy=self.policy, nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                    enable_double_dqn=True, enable_dueling_network=True)
        
        # Compile the agent with the adam optimizer and with the mean absolute error metric
        self.agent.compile(Adam(lr=1e-3), metrics=['mae'])

        # Save the weights of the agents in the q.weights file
        # Save random weights
        self.agent.save_weights("q.weights", overwrite=True)

        # Define the current starting point as the initial date
        self.currentStartingPoint = begin
        
        # Define the training, validation and testing size as informed by the call
        # Train: 5 years
        # Validation: 6 months
        # Test: 6 months
        self.trainSize = trainSize
        self.validationSize = validationSize
        self.testSize = testSize
        
        # The walk size is simply summing up the train, validation and test sizes
        self.walkSize = trainSize + validationSize + testSize
        
        # Define the ending point as the final date (January 1st of 2010)
        self.endingPoint = end

        # Read the hourly dataset
        # We join data from different files
        # Here hour data is read 
        self.dates = pd.read_csv('datasets/'+ticker+'hour.csv')
        self.sp = pd.read_csv('datasets/'+ticker+'hour.csv')
        # Convert the pandas format to date and time format
        self.sp['Datetime'] = pd.to_datetime(self.sp['Date'] + ' ' + self.sp['Time'])
        # Set an index to Datetime on the pandas loaded dataset. Registers will be indexes through these values
        self.sp = self.sp.set_index('Datetime')
        # Drop Time and Date from the Dataset
        self.sp = self.sp.drop(['Time','Date'], axis=1)
        # Just the index considering date and time will be important, because date and time will be used to define the train, 
        # validation and test for each walk
        self.sp = self.sp.index

        # Receives the operation cost, which is 0
        # Operation cost is the cost for long and short. It is defined as zero
        self.operationCost = operationCost
        
        # Call the callback for training, validation and test in order to show results for each episode 
        self.trainer = ValidationCallback()
        self.validator = ValidationCallback()
        self.tester = ValidationCallback()
        self.outputFileName = outputFile

    def run(self):
    
        # Initiates the environments
        trainEnv = validEnv = testEnv = " "
        iteration = -1

        # While we did not pass through all the dates (i.e., while all the walks were not finished)
        # walk size is train+validation+test size
        # currentStarting point begins with begin date
        self.numwalks = 0
        while(self.currentStartingPoint + self.walkSize <= self.endingPoint):

            # Iteration is the current walk
            iteration+=1

            # Initiate the output file
            self.outputFile = open(self.outputFileName+str(iteration+1)+f"{self.ticker}.csv", "w+")
            # write the first row of the csv
            self.outputFile.write(
                "Iteration,"+
                "trainAccuracy,"+
                "trainCoverage,"+
                "trainReward,"+
                "trainLong%,"+
                "trainShort%,"+
                "trainLongAcc,"+
                "trainShortAcc,"+
                "trainLongPrec,"+
                "trainShortPrec,"+

                "validationAccuracy,"+
                "validationCoverage,"+
                "validationReward,"+
                "validationLong%,"+
                "validationShort%,"+
                "validationLongAcc,"+
                "validationShortAcc,"+
                "validLongPrec,"+
                "validShortPrec,"+
                
                "testAccuracy,"+
                "testCoverage,"+
                "testReward,"+
                "testLong%,"+
                "testShort%,"+
                "testLongAcc,"+
                "testShortAcc,"+
                "testLongPrec,"+
                "testShortPrec\n")


            
            # Empty the memory and agent
            del(self.memory)
            del(self.agent)

            # Define the memory and agent
            # Memory is Sequential
            self.memory = SequentialMemory(limit=10000, window_length=1)

            # Agent is initiated as passed through parameters
            self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1,
                                    enable_double_dqn=True,enable_dueling_network=True)
            
            # Compile the agent with Adam initialization
            self.agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
            
            # Load the weights saved before in a random way if it is the first time
            self.agent.load_weights("q.weights")
            
            ########################################TRAINING STAGE########################################################
            
            # The TrainMinLimit will be loaded as the initial date at the beginning, and will be updated later.
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date    
            trainMinLimit = None
            while(trainMinLimit is None):
                try:
                    trainMinLimit = self.sp.get_loc(self.currentStartingPoint)
                except:
                    # datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
                    self.currentStartingPoint+=datetime.timedelta(0, 0, 0, 0, 30, 0, 0)

            # The TrainMaxLimit will be loaded as the interval between the initial date plus the training size.
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date    
            trainMaxLimit=None
            while(trainMaxLimit is None):
                try:
                    trainMaxLimit = self.sp.get_loc(self.currentStartingPoint + self.trainSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0, 0, 0, 0, 30, 0, 0)
            
            ########################################VALIDATION STAGE#######################################################
            # The ValidMinLimit will be loaded as the next element of the TrainMax limit
            validMinLimit = trainMaxLimit+1

            # The ValidMaxLimit will be loaded as the interval after the begin + train size +validation size
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date  
            validMaxLimit = None
            while(validMaxLimit is None):
                try:
                    validMaxLimit = self.sp.get_loc(self.currentStartingPoint + self.trainSize + self.validationSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0, 0, 0, 0, 30, 0, 0)

            ########################################TESTING STAGE######################################################## 
            # The TestMinLimit will be loaded as the next element of ValidMaxlimit 
            testMinLimit = validMaxLimit+1

            # The testMaxLimit will be loaded as the interval after the begin + train size +validation size + Testsize
            # If the initial date cannot be used, add 1 hour to the initial date and consider it the initial date 
            testMaxLimit = None
            while(testMaxLimit is None):
                try:
                    testMaxLimit = self.sp.get_loc(self.currentStartingPoint + self.trainSize + self.validationSize + self.testSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0, 0, 0, 0, 30, 0, 0)

            # Separate the Validation and testing data according to the limits found before
            # Prepare the training and validation files for saving them later 
            ensambleValid = pd.DataFrame(index=self.dates[validMinLimit:validMaxLimit].loc[:,'Date'].drop_duplicates().tolist())
            ensambleTest = pd.DataFrame(index=self.dates[testMinLimit:testMaxLimit].loc[:,'Date'].drop_duplicates().tolist())
            
            # Put the name of the index for validation and testing
            ensambleValid.index.name = 'Date'
            ensambleTest.index.name = 'Date'
            
            # Explorations are epochs considered, or how many times the agent will play the game.  
            for eps in self.explorations:

                # policy will be 0.2, so the randomness of predictions (actions) will happen with 20% of probability 
                self.policy.eps = eps[0]
                
                # there will be 100 iterations (epochs), or eps[1])
                for i in range(0, eps[1]):
                    
                    del(trainEnv)

                    # Define the training, validation and testing environments with their respective callbacks
                    trainEnv = SpEnv(ticker=self.ticker,operationCost=self.operationCost,minLimit=trainMinLimit,maxLimit=trainMaxLimit,callback=self.trainer,isOnlyShort=self.isOnlyShort)
                    del(validEnv)
                    validEnv = SpEnv(ticker=self.ticker,operationCost=self.operationCost,minLimit=validMinLimit,maxLimit=validMaxLimit,callback=self.validator,isOnlyShort=self.isOnlyShort,ensamble=ensambleValid,columnName="iteration"+str(i))
                    del(testEnv)
                    testEnv = SpEnv(ticker=self.ticker,operationCost=self.operationCost,minLimit=testMinLimit,maxLimit=testMaxLimit,callback=self.tester,isOnlyShort=self.isOnlyShort,ensamble=ensambleTest,columnName="iteration"+str(i))

                    # Reset the callback
                    self.trainer.reset()
                    self.validator.reset()
                    self.tester.reset()

                    # Reset the training environment
                    trainEnv.resetEnv()
                    # Train the agent
                    self.agent.fit(trainEnv, nb_steps=floor(self.trainSize.days-self.trainSize.days*0.2),visualize=False,verbose=0)
                    # Get the info from the train callback
                    (_,trainCoverage,trainAccuracy,trainReward,trainLongPerc,trainShortPerc,trainLongAcc,trainShortAcc,trainLongPrec,trainShortPrec)=self.trainer.getInfo()
                    # Print Callback values on the screen
                    print(str(i) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward))

                    # Reset the validation environment
                    validEnv.resetEnv()
                    # Test the agent on validation data
                    self.agent.test(validEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                    # Get the info from the validation callback
                    (_,validCoverage,validAccuracy,validReward,validLongPerc,validShortPerc,validLongAcc,validShortAcc,validLongPrec,validShortPrec)=self.validator.getInfo()
                    # Print callback values on the screen
                    print(str(i) + " VALID:  acc: " + str(validAccuracy)+ " cov: " + str(validCoverage)+ " rew: " + str(validReward))

                    # Reset the testing environment
                    testEnv.resetEnv()
                    # Test the agent on testing data
                    self.agent.test(testEnv,nb_episodes=floor(self.validationSize.days-self.validationSize.days*0.2),visualize=False,verbose=0)
                    # Get the info from the testing callback
                    (_,testCoverage,testAccuracy,testReward,testLongPerc,testShortPerc,testLongAcc,testShortAcc,testLongPrec,testShortPrec)=self.tester.getInfo()
                    # Print callback values on the screen
                    print(str(i) + " TEST:  acc: " + str(testAccuracy)+ " cov: " + str(testCoverage)+ " rew: " + str(testReward))
                    print(" ")
                    
                    # write the walk data on the text file
                    self.outputFile.write(
                        str(i)+","+
                        str(trainAccuracy)+","+
                        str(trainCoverage)+","+
                        str(trainReward)+","+
                        str(trainLongPerc)+","+
                        str(trainShortPerc)+","+
                        str(trainLongAcc)+","+
                        str(trainShortAcc)+","+
                        str(trainLongPrec)+","+
                        str(trainShortPrec)+","+
                        
                        str(validAccuracy)+","+
                        str(validCoverage)+","+
                        str(validReward)+","+
                        str(validLongPerc)+","+
                        str(validShortPerc)+","+
                        str(validLongAcc)+","+
                        str(validShortAcc)+","+
                        str(validLongPrec)+","+
                        str(validShortPrec)+","+
                        
                        str(testAccuracy)+","+
                        str(testCoverage)+","+
                        str(testReward)+","+
                        str(testLongPerc)+","+
                        str(testShortPerc)+","+
                        str(testLongAcc)+","+
                        str(testShortAcc)+","+
                        str(testLongPrec)+","+
                        str(testShortPrec)+"\n")

            # Close the file                
            self.outputFile.close()

            # For the next walk, the current starting point will be the current starting point + the test size
            # It means that, for the next walk, the training data will start 6 months after the training data of 
            # the previous walk   
            self.currentStartingPoint += self.testSize

            # Write validation and Testing data into files
            # Save the files for processing later with the ensemble considering the 100 epochs
            ensambleValid.to_csv("Output/ensemble/"+self.ensembleFolderName+"/walk"+str(iteration)+f"ensemble_valid_{self.ticker}.csv")
            ensambleTest.to_csv("Output/ensemble/"+self.ensembleFolderName+"/walk"+str(iteration)+f"ensemble_test_{self.ticker}.csv")
            self.numwalks += 1
        return self.numwalks
    # Function to end the Agent
    def end(self):
        print("END")