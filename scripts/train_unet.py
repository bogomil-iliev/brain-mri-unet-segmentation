# Fitting of the model
K.clear_session()

history =  model.fit(training_generator,
                    epochs=100,
                    #steps_per_epoch=len(train_ids),
                    callbacks= callbacks,
                    validation_data = valid_generator
                    )

# Save the trained model
model.save("my_model_100.keras")
