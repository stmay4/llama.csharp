using Xunit;
using FluentAssertions;
using Llama.csharp;

namespace Llama.csharp.UnitTests
{
    [Trait("Category", "Unit")]
    [Trait("Category", "Samplers")]
    public class TestSamplersConstructor
    {
        #region TopKSampler

        [Theory]
        [InlineData(-1)]
        [InlineData(-100)]
        public void TopKSampler_K_Negative_ThrowsArgumentOutOfRangeException(int invalidK)
        {
            var act = () => new TopKSampler { K = invalidK };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*K*");
        }

        [Theory]
        [InlineData(0)]      // граница: разрешено
        [InlineData(1)]
        [InlineData(100)]
        public void TopKSampler_K_Valid_SetProperty(int validK)
        {
            var sampler = new TopKSampler { K = validK };
            sampler.K.Should().Be(validK);
        }

        [Fact]
        public void TopKSampler_K_Default_Is40()
        {
            var sampler = new TopKSampler();
            sampler.K.Should().Be(40);
        }

        #endregion

        #region TopPSampler

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(1.1f)]
        public void Constructor_InvalidP_ThrowsArgumentOutOfRangeException(float invalidP)
        {
            var act = () => new TopPSampler { P = invalidP };

            act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*P*"); ;
        }

        [Fact]
        public void Constructor_ValidP_SetProperty()
        {
            var expectedP = 0.9f;

            var sampler = new TopPSampler { P = expectedP };

            sampler.P.Should().Be(expectedP);
        }

        [Fact]
        public void Constructor_DefaultValues_AreCorrect()
        {
            float defaultP = 0.95f;
            int minKeep = 1;
            var sampler = new TopPSampler();

            sampler.P.Should().Be(defaultP);
            sampler.MinKeep.Should().Be(minKeep);
        }
        #endregion

        #region MinPSampler

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(1.1f)]
        public void MinPSampler_P_Invalid_ThrowsArgumentOutOfRangeException(float invalidP)
        {
            var act = () => new MinPSampler { P = invalidP };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*P*");
        }

        [Theory]
        [InlineData(0f)]     // граница: разрешено
        [InlineData(0.5f)]
        [InlineData(1f)]     // граница: разрешено
        public void MinPSampler_P_Valid_SetProperty(float validP)
        {
            var sampler = new MinPSampler { P = validP };
            sampler.P.Should().Be(validP);
        }

        [Fact]
        public void MinPSampler_P_Default_Is0_05()
        {
            var sampler = new MinPSampler();
            sampler.P.Should().Be(0.05f);
        }

        [Theory]
        [InlineData(-1)]
        public void MinPSampler_MinKeep_Negative_ThrowsArgumentOutOfRangeException(int invalidMinKeep)
        {
            var act = () => new MinPSampler { MinKeep = invalidMinKeep };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*MinKeep*");
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(100)]
        public void MinPSampler_MinKeep_Valid_SetProperty(int validMinKeep)
        {
            var sampler = new MinPSampler { MinKeep = validMinKeep };
            sampler.MinKeep.Should().Be(validMinKeep);
        }

        [Fact]
        public void MinPSampler_MinKeep_Default_Is1()
        {
            var sampler = new MinPSampler();
            sampler.MinKeep.Should().Be(1);
        }

        #endregion

        #region TypicalSampler

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(1.1f)]
        public void TypicalSampler_P_Invalid_ThrowsArgumentOutOfRangeException(float invalidP)
        {
            var act = () => new TypicalSampler { P = invalidP };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*P*");
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(0.2f)]
        [InlineData(1f)]
        public void TypicalSampler_P_Valid_SetProperty(float validP)
        {
            var sampler = new TypicalSampler { P = validP };
            sampler.P.Should().Be(validP);
        }

        [Fact]
        public void TypicalSampler_P_Default_Is0_2()
        {
            var sampler = new TypicalSampler();
            sampler.P.Should().Be(0.2f);
        }

        [Theory]
        [InlineData(-1)]
        public void TypicalSampler_MinKeep_Negative_ThrowsArgumentOutOfRangeException(int invalidMinKeep)
        {
            var act = () => new TypicalSampler { MinKeep = invalidMinKeep };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*MinKeep*");
        }

        [Fact]
        public void TypicalSampler_MinKeep_Default_Is1()
        {
            var sampler = new TypicalSampler();
            sampler.MinKeep.Should().Be(1);
        }

        #endregion

        #region TemperatureSampler

        [Theory]
        [InlineData(0f)]      // граница: запрещено (строго > 0)
        [InlineData(-0.1f)]
        public void TemperatureSampler_T_Invalid_ThrowsArgumentOutOfRangeException(float invalidT)
        {
            var act = () => new TemperatureSampler { T = invalidT };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*T*");
        }

        [Theory]
        [InlineData(0.001f)]
        [InlineData(0.8f)]
        [InlineData(2.0f)]
        public void TemperatureSampler_T_Valid_SetProperty(float validT)
        {
            var sampler = new TemperatureSampler { T = validT };
            sampler.T.Should().Be(validT);
        }

        [Fact]
        public void TemperatureSampler_T_Default_Is0_8()
        {
            var sampler = new TemperatureSampler();
            sampler.T.Should().Be(0.8f);
        }

        #endregion

        #region AdapTSampler

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(1.1f)]
        public void AdapTSampler_T_Invalid_ThrowsArgumentOutOfRangeException(float invalidT)
        {
            var act = () => new AdapTSampler { T = invalidT };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*T*");
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(0.5f)]
        [InlineData(1f)]
        public void AdapTSampler_T_Valid_SetProperty(float validT)
        {
            var sampler = new AdapTSampler { T = validT };
            sampler.T.Should().Be(validT);
        }

        [Fact]
        public void AdapTSampler_T_Default_Is0_8()
        {
            var sampler = new AdapTSampler();
            sampler.T.Should().Be(0.8f);
        }

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(1f)]        // граница: запрещено (строго < 1)
        [InlineData(1.1f)]
        public void AdapTSampler_Delta_Invalid_ThrowsArgumentOutOfRangeException(float invalidDelta)
        {
            var act = () => new AdapTSampler { Delta = invalidDelta };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*Delta*");
        }

        [Theory]
        [InlineData(0f)]        // граница: разрешено
        [InlineData(0.4f)]
        [InlineData(0.99f)]     // близко к границе
        public void AdapTSampler_Delta_Valid_SetProperty(float validDelta)
        {
            var sampler = new AdapTSampler { Delta = validDelta };
            sampler.Delta.Should().Be(validDelta);
        }

        [Fact]
        public void AdapTSampler_Delta_Default_Is0_4()
        {
            var sampler = new AdapTSampler();
            sampler.Delta.Should().Be(0.40f);
        }

        [Fact]
        public void AdapTSampler_Exponent_Default_Is1()
        {
            var sampler = new AdapTSampler();
            sampler.Exponent.Should().Be(1f);
        }

        #endregion

        #region XTCSampler

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(1.1f)]
        public void XTCSampler_P_Invalid_ThrowsArgumentOutOfRangeException(float invalidP)
        {
            var act = () => new XTCSampler { P = invalidP };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*P*");
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(0.5f)]
        [InlineData(1f)]
        public void XTCSampler_P_Valid_SetProperty(float validP)
        {
            var sampler = new XTCSampler { P = validP };
            sampler.P.Should().Be(validP);
        }

        [Fact]
        public void XTCSampler_P_Default_Is0_5()
        {
            var sampler = new XTCSampler();
            sampler.P.Should().Be(0.5f);
        }

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(1.1f)]
        public void XTCSampler_T_Invalid_ThrowsArgumentOutOfRangeException(float invalidT)
        {
            var act = () => new XTCSampler { T = invalidT };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*T*");
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(0.5f)]
        [InlineData(1f)]
        public void XTCSampler_T_Valid_SetProperty(float validT)
        {
            var sampler = new XTCSampler { T = validT };
            sampler.T.Should().Be(validT);
        }

        [Fact]
        public void XTCSampler_T_Default_Is0_1()
        {
            var sampler = new XTCSampler();
            sampler.T.Should().Be(0.1f);
        }

        [Theory]
        [InlineData(-1)]
        public void XTCSampler_MinKeep_Invalid_ThrowsArgumentOutOfRangeException(int invalidMinKeep)
        {
            var act = () => new XTCSampler { MinKeep = invalidMinKeep };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*MinKeep*");
        }

        [Theory]
        [InlineData(1)]
        [InlineData(4)]
        [InlineData(10)]
        public void XTCSampler_MinKeep_Valid_SetProperty(int validMinKeep)
        {
            var sampler = new XTCSampler { MinKeep = validMinKeep };
            sampler.MinKeep.Should().Be(validMinKeep);
        }

        [Fact]
        public void XTCSampler_MinKeep_Default_Is1()
        {
            var sampler = new XTCSampler();
            sampler.MinKeep.Should().Be(1);
        }

        #endregion

        #region PenaltiesSampler

        [Theory]
        [InlineData(-2.1f)]
        [InlineData(2.1f)]
        public void PenaltiesSampler_FrequencyPenalty_Invalid_ThrowsArgumentOutOfRangeException(float invalidValue)
        {
            var act = () => new PenaltiesSampler { FrequencyPenalty = invalidValue };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*FrequencyPenalty*");
        }

        [Theory]
        [InlineData(-2f)]       // граница: разрешено
        [InlineData(0f)]
        [InlineData(2f)]        // граница: разрешено
        public void PenaltiesSampler_FrequencyPenalty_Valid_SetProperty(float validValue)
        {
            var sampler = new PenaltiesSampler { FrequencyPenalty = validValue };
            sampler.FrequencyPenalty.Should().Be(validValue);
        }

        [Fact]
        public void PenaltiesSampler_FrequencyPenalty_Default_Is0()
        {
            var sampler = new PenaltiesSampler();
            sampler.FrequencyPenalty.Should().Be(0f);
        }

        [Theory]
        [InlineData(-2.1f)]
        [InlineData(2.1f)]
        public void PenaltiesSampler_PresencePenalty_Invalid_ThrowsArgumentOutOfRangeException(float invalidValue)
        {
            var act = () => new PenaltiesSampler { PresencePenalty = invalidValue };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*PresencePenalty*");
        }

        [Theory]
        [InlineData(-2f)]
        [InlineData(0f)]
        [InlineData(2f)]
        public void PenaltiesSampler_PresencePenalty_Valid_SetProperty(float validValue)
        {
            var sampler = new PenaltiesSampler { PresencePenalty = validValue };
            sampler.PresencePenalty.Should().Be(validValue);
        }

        [Fact]
        public void PenaltiesSampler_PresencePenalty_Default_Is0()
        {
            var sampler = new PenaltiesSampler();
            sampler.PresencePenalty.Should().Be(0f);
        }

        [Fact]
        public void PenaltiesSampler_DefaultValues_AreCorrect()
        {
            var sampler = new PenaltiesSampler();
            sampler.PenaltyCount.Should().Be(64);
            sampler.RepeatPenalty.Should().Be(1f);
        }

        #endregion

        #region Mirostat2Sampler

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(-10f)]
        public void Mirostat2Sampler_Tau_Invalid_ThrowsArgumentOutOfRangeException(float invalidTau)
        {
            var act = () => new Mirostat2Sampler { Tau = invalidTau };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*Tau*");
        }

        [Theory]
        [InlineData(0f)]        // граница: разрешено
        [InlineData(1f)]
        [InlineData(10f)]
        public void Mirostat2Sampler_Tau_Valid_SetProperty(float validTau)
        {
            var sampler = new Mirostat2Sampler { Tau = validTau };
            sampler.Tau.Should().Be(validTau);
        }

        [Fact]
        public void Mirostat2Sampler_Tau_Default_Is3()
        {
            var sampler = new Mirostat2Sampler();
            sampler.Tau.Should().Be(3f);
        }

        [Theory]
        [InlineData(-0.1f)]
        [InlineData(-1f)]
        public void Mirostat2Sampler_Eta_Invalid_ThrowsArgumentOutOfRangeException(float invalidEta)
        {
            var act = () => new Mirostat2Sampler { Eta = invalidEta };
            act.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("*Eta*");
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(0.1f)]
        [InlineData(5f)]
        public void Mirostat2Sampler_Eta_Valid_SetProperty(float validEta)
        {
            var sampler = new Mirostat2Sampler { Eta = validEta };
            sampler.Eta.Should().Be(validEta);
        }

        [Fact]
        public void Mirostat2Sampler_Eta_Default_Is0_1()
        {
            var sampler = new Mirostat2Sampler();
            sampler.Eta.Should().Be(0.1f);
        }

        [Fact]
        public void Mirostat2Sampler_Seed_Default_Is42()
        {
            var sampler = new Mirostat2Sampler();
            sampler.Seed.Should().Be(42u);
        }

        #endregion
    }
}
